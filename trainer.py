import logging
import time
import math
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from dataset import TransformOpenMatch, cifar10_mean, cifar10_std, \
    cifar100_std, cifar100_mean, normal_mean, \
    normal_std, TransformFixMatch_Imagenet_Weak
from tqdm import tqdm
from utils import AverageMeter, save_checkpoint, test, test_ood, exclude_dataset
from loss_funtion import *

logger = logging.getLogger(__name__)
best_acc = 0
best_acc_val = 0

def train(args, labeled_trainloader, unlabeled_dataset, test_loader, val_loader, ood_loaders,
          models, optimizers, ema_model, schedulers):
    # if args.amp:
    #     from apex import amp

    global best_acc
    global best_acc_val

    test_accs = []
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_cls = AverageMeter()
    losses_ood = AverageMeter()
    losses_ssl = AverageMeter()
    losses_rec_c = AverageMeter()
    losses_dom = AverageMeter()
    losses_rec_s = AverageMeter()
    mask_probs = AverageMeter()
    losses_dis_c = AverageMeter()
    losses_dis_s = AverageMeter()
    losses_vae_c = AverageMeter()
    losses_vae_s = AverageMeter()
    losses_aug = AverageMeter()

    end = time.time()
    entr_factor = 1 + math.log(args.num_classes / args.aug_num)
    augmentation_flag = False

    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
    labeled_iter = iter(labeled_trainloader)
    default_out = "Epoch:{epoch}/{epochs:4}. " \
                  "LR:{lr:.6f}. " \
                  "SUP:{losses_cls:.4f} " \
                  "OOD:{losses_ood:.4f}"
    output_args = vars(args)
    default_out += " SSL:{losses_ssl:.4f}"
    default_out += " DOM:{losses_dom:.4f}"
    default_out += " RECc:{losses_rec_c:.4f}"
    default_out += " DISc:{losses_dis_c:.4f}"
    default_out += " VAEc:{losses_vae_c:.4f}"
    default_out += " AUG:{losses_aug:.4f}"

    model_c, model_s = models
    model_c.train()
    model_s.train()

    optimizer_c, optimizer_s = optimizers
    scheduler_c, scheduler_s = schedulers

    unlabeled_dataset_all = copy.deepcopy(unlabeled_dataset)
    if args.dataset == 'cifar10':
        mean = cifar10_mean
        std = cifar10_std
        func_trans = TransformOpenMatch
    elif args.dataset == 'cifar100':
        mean = cifar100_mean
        std = cifar100_std
        func_trans = TransformOpenMatch
    elif 'imagenet' in args.dataset:
        mean = normal_mean
        std = normal_std
        func_trans = TransformFixMatch_Imagenet_Weak

    
    # rewritten our TransformHOOD to TransformOpenMatch which only conducts two weak transformations
    unlabeled_dataset_all.transform = func_trans(mean=mean, std=std)
    labeled_dataset_all = copy.deepcopy(labeled_trainloader.dataset)
    labeled_dataset_all.transform = func_trans(mean=mean, std=std)
    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler
    labeled_trainloader_all = DataLoader(
        labeled_dataset_all,
        sampler=train_sampler(labeled_dataset_all),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True)

    for epoch in range(args.start_epoch, args.epochs):
        
        output_args["epoch"] = epoch
        if not args.no_progress:
            p_bar = tqdm(range(args.eval_step),
                         disable=args.local_rank not in [-1, 0])

        if epoch >= args.start_fix:
            ## pick pseudo-inliers
            exclude_dataset(args, unlabeled_dataset, ema_model.ema)
        if epoch >= args.start_augmentation:
            if args.augment is True:
                augmentation_flag = True

        unlabeled_trainloader = DataLoader(unlabeled_dataset,
                                           sampler = train_sampler(unlabeled_dataset),
                                           batch_size = args.batch_size * args.mu,
                                           num_workers = args.num_workers,
                                           drop_last = True)
        unlabeled_trainloader_all = DataLoader(unlabeled_dataset_all,
                                           sampler=train_sampler(unlabeled_dataset_all),
                                           batch_size=args.batch_size * args.mu,
                                           num_workers=args.num_workers,
                                           drop_last=True)

        unlabeled_iter = iter(unlabeled_trainloader)
        unlabeled_all_iter = iter(unlabeled_trainloader_all)

        for batch_idx in range(args.eval_step):            
            # HOOD augmentation
            try:
                l_inputs, targets_x = labeled_iter.next()
            except:
                if args.world_size > 1:
                    labeled_epoch += 1
                    labeled_trainloader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(labeled_trainloader)
                l_inputs, targets_x = labeled_iter.next()
            try:
                u_inputs, _ = unlabeled_iter.next()
            except:
                if args.world_size > 1:
                    unlabeled_epoch += 1
                    unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(unlabeled_trainloader)
                u_inputs, _ = unlabeled_iter.next()
            # weak augmentation
            try:
                l_inputs_all, targets_x_all = labeled_iter_all.next()
            except:
                labeled_iter_all = iter(labeled_trainloader_all)
                l_inputs_all, targets_x_all = labeled_iter_all.next()
            try:
                u_inputs_all, _ = unlabeled_all_iter.next()
            except:
                unlabeled_all_iter = iter(unlabeled_trainloader_all)
                u_inputs_all, _ = unlabeled_all_iter.next()


            data_time.update(time.time() - end)
            b_size = l_inputs_all[0].shape[0]

            inputs_weak = torch.cat([l_inputs_all[0], l_inputs_all[1], u_inputs_all[0], \
                                u_inputs_all[1]], 0).to(args.device)

            targets_x = targets_x.to(args.device)
            targets_x_all = targets_x_all.to(args.device)

            # forward pass
            cls_w, cls_open_w, content, cls_rec, cls_mu, cls_logvar = model_c(inputs_weak, stats=True)
            _, _, style, dom_rec, dom_mu, dom_logvar = model_s(inputs_weak, stats=True)

            # compute loss
            loss_rec_c, loss_rec_s = cal_reconstruction(args, cls_rec, dom_rec, inputs_weak)
            loss_ood = cal_ood(args, cls_open_w, targets_x_all.repeat(2), b_size)
            loss_cls = cal_content_classification(cls_w[:2*b_size], targets_x_all.repeat(2))
            vae_cls = args.lambda_vae * cal_vae_loss(cls_mu, cls_logvar)
            vae_dom = args.lambda_vae * cal_vae_loss(dom_mu, dom_logvar)

            # HOOD augmented data
            inputs_hood = torch.cat([l_inputs[0], l_inputs[1], u_inputs[0], u_inputs[1]], 0).to(args.device)
            aug_idx = l_inputs[2].to(args.device).long()

            cls_s, _, content, _ = model_c(inputs_hood)
            dom_s, _, style, _ = model_s(inputs_hood)

            # compute loss for HOOD augmented data
            loss_dom = cal_style_classification(dom_s[b_size:2*b_size], aug_idx)

            if epoch >= args.start_fix:
                loss_ssl = cal_ssl(args, cls_s[2*b_size:], mask_probs)

            else:
                loss_ssl = torch.zeros(1).to(args.device).mean()

            if args.disentangle:
                content_disent, style_disent = model_s.disentangle(content[b_size:2*b_size]), model_c.disentangle(style[b_size:2*b_size])
                disent_cls = -args.lambda_disent * cal_entropy(content_disent) # negative entropy
                disent_dom = -args.lambda_disent * cal_entropy(style_disent) # negative entropy
            else:
                disent_cls = torch.zeros(1).to(args.device).mean()
                disent_dom = torch.zeros(1).to(args.device).mean()

            loss_c = loss_ood + loss_cls + loss_ssl + loss_rec_c + vae_cls + disent_cls
            loss_s = loss_dom + loss_rec_s + vae_dom + disent_dom

            if augmentation_flag and epoch % args.augmentation_interval < args.augmentation_round:
                to_aug_inputs = torch.cat([l_inputs_all[0], l_inputs_all[1]], 0).to(args.device)
                benign_inputs = augmentation(args, mean, std, model_c, model_s, to_aug_inputs, targets_x_all.repeat(2), aug_type='benign', domain_targeted=False)
                malign_inputs = augmentation(args, mean, std, model_c, model_s, to_aug_inputs, targets_x_all.repeat(2), aug_type='malign', domain_targeted=False)
                
                aug_inputs = torch.cat([benign_inputs, malign_inputs], 0).to(args.device)
                cls_aug, cls_aug_open, _, _ = model_c(aug_inputs)
                cls_aug_benign, _ = cls_aug.chunk(2)
                _, cls_aug_open_malign = cls_aug_open.chunk(2)
                loss_aug = cal_content_classification(cls_aug_benign, targets_x_all.repeat(2))
                loss_aug += cal_ood(args, cls_aug_open_malign, targets_x_all.repeat(2), negative=True)
                loss_c += args.lambda_aug * loss_aug
            else:
                loss_aug = torch.zeros(1).to(args.device).mean()

            # if args.amp:
            #     with amp.scale_loss(loss, optimizer) as scaled_loss:
                    # scaled_loss.bacbatch_idxward()
            
            losses.update(loss_c.item())
            losses_cls.update(loss_cls.item())
            losses_ood.update(loss_ood.item())
            losses_ssl.update(loss_ssl.item())
            losses_dis_c.update(disent_cls.item())
            losses_rec_c.update(loss_rec_c.item())
            losses_dom.update(loss_dom.item())
            losses_dis_s.update(disent_dom.item())
            losses_rec_s.update(loss_rec_s.item())
            losses_vae_c.update(vae_cls.item())
            losses_vae_s.update(vae_dom.item())
            losses_aug.update(loss_aug.item())

            output_args["batch"] = batch_idx
            output_args["losses_cls"] = losses_cls.avg
            output_args["losses_ood"] = losses_ood.avg
            output_args["losses_ssl"] = losses_ssl.avg
            output_args["losses_dom"] = losses_dom.avg
            output_args["losses_rec_c"] = losses_rec_c.avg
            output_args["losses_rec_s"] = losses_rec_s.avg
            output_args["losses_dis_c"] = losses_dis_c.avg
            output_args["losses_dis_s"] = losses_dis_s.avg
            output_args["losses_vae_c"] = losses_vae_c.avg
            output_args["losses_vae_s"] = losses_vae_s.avg
            output_args["losses_aug"] = losses_aug.avg
    
            output_args["lr"] = [group["lr"] for group in optimizer_c.param_groups][0]

            optimizer_c.zero_grad()
            loss_c.backward()
            optimizer_c.step()
            
            optimizer_s.zero_grad()
            loss_s.backward()
            optimizer_s.step()


            if args.opt != 'adam':
                scheduler_c.step()
                scheduler_s.step()
            if args.use_ema:
                ema_model.update(model_c)
            model_c.zero_grad()
            batch_time.update(time.time() - end)
            end = time.time()

            if not args.no_progress:
                p_bar.set_description(default_out.format(**output_args))
                p_bar.update()

        if not args.no_progress:
            p_bar.close()

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model_c

        if args.local_rank in [-1, 0]:

            val_acc = test(args, val_loader, test_model, epoch, val=True)
            test_loss, test_acc_close, test_overall, \
            test_unk, test_roc, test_roc_softm, test_id \
                = test(args, test_loader, test_model, epoch)

            for ood in ood_loaders.keys():
                roc_ood = test_ood(args, test_id, ood_loaders[ood], test_model)
                logger.info("ROC vs {ood}: {roc}".format(ood=ood, roc=roc_ood))

            args.writer.add_scalar('train/1.train_loss', losses.avg, epoch)
            args.writer.add_scalar('train/2.train_loss_cls', losses_cls.avg, epoch)
            args.writer.add_scalar('train/3.train_loss_ood', losses_ood.avg, epoch)
            args.writer.add_scalar('train/4.train_loss_ssl', losses_ssl.avg, epoch)
            args.writer.add_scalar('train/5.train_loss_dom', losses_dom.avg, epoch)
            args.writer.add_scalar('train/6.train_loss_rec_c', losses_rec_c.avg, epoch)
            args.writer.add_scalar('train/7.train_loss_rec_s', losses_rec_s.avg, epoch)
            args.writer.add_scalar('train/8.train_loss_dis_c', losses_dis_c.avg, epoch)
            args.writer.add_scalar('train/9.train_loss_dis_s', losses_dis_s.avg, epoch)
            args.writer.add_scalar('train/10.mask', mask_probs.avg, epoch)
            args.writer.add_scalar('test/1.test_acc', test_acc_close, epoch)
            args.writer.add_scalar('test/2.test_loss', test_loss, epoch)

            is_best = val_acc > best_acc_val
            best_acc_val = max(val_acc, best_acc_val)
            if is_best:
                overall_valid = test_overall
                close_valid = test_acc_close
                unk_valid = test_unk
                roc_valid = test_roc
                roc_softm_valid = test_roc_softm
            model_to_save_c = model_c.module if hasattr(model_c, "module") else model_c
            model_to_save_s = model_s.module if hasattr(model_s, "module") else model_s
            if args.use_ema:
                ema_to_save = ema_model.ema.module if hasattr(
                    ema_model.ema, "module") else ema_model.ema

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict_c': model_to_save_c.state_dict(),
                'state_dict_s': model_to_save_s.state_dict(),
                'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                'acc close': test_acc_close,
                'acc overall': test_overall,
                'unk': test_unk,
                'best_acc': best_acc,
                'optimizer_c': optimizer_c.state_dict(),
                'scheduler_c': scheduler_c.state_dict(),
                'optimizer_s': optimizer_s.state_dict(),
                'scheduler_s': scheduler_s.state_dict(),
            }, is_best, args.out)
            test_accs.append(test_acc_close)
            logger.info('Best val closed acc: {:.3f}'.format(best_acc_val))
            logger.info('Valid closed acc: {:.3f}'.format(close_valid))
            logger.info('Valid overall acc: {:.3f}'.format(overall_valid))
            logger.info('Valid unk acc: {:.3f}'.format(unk_valid))
            logger.info('Valid roc: {:.3f}'.format(roc_valid))
            logger.info('Valid roc soft: {:.3f}'.format(roc_softm_valid))
            logger.info('Mean top-1 acc: {:.3f}\n'.format(
                np.mean(test_accs[-20:])))
    if args.local_rank in [-1, 0]:
        args.writer.close()
