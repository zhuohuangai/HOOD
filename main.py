import logging
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from utils import set_model_config, \
    set_dataset, set_models, set_parser, \
    set_seed
from eval import eval_model
from trainer import train

logger = logging.getLogger(__name__)


def main():
    args = set_parser()
    global best_acc
    global best_acc_val

    if args.local_rank == -1:
        device = torch.device('cuda', args.gpu_id)
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
        args.n_gpu = 1
    args.device = device
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning(
        f"Process rank: {args.local_rank}, "
        f"device: {args.device}, "
        f"n_gpu: {args.n_gpu}, "
        f"distributed training: {bool(args.local_rank != -1)}, "
        f"16-bits training: {args.amp}",)
    logger.info(dict(args._get_kwargs()))
    if args.seed is not None:
        set_seed(args)
    if args.local_rank in [-1, 0]:
        os.makedirs(args.out, exist_ok=True)
        args.writer = SummaryWriter(args.out)
    set_model_config(args)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    labeled_trainloader, unlabeled_dataset, test_loader, val_loader, ood_loaders\
        = set_dataset(args)
    
    models, optimizers, schedulers = set_models(args)
    model_c, model_s = models
    optimizer_c, optimizer_s = optimizers
    scheduler_c, scheduler_s = schedulers
    
    logger.info("Total params: {:.2f}M".format(
        2 * sum(p.numel() for p in model_c.parameters()) / 1e6))

    if args.use_ema:
        from models.ema import ModelEMA
        ema_model = ModelEMA(args, model_c, args.ema_decay)
    args.start_epoch = 0
    if args.resume:
        logger.info("==> Resuming from checkpoint..")
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        # best_acc = checkpoint['best_acc']
        # args.start_epoch = checkpoint['epoch']
        model_c.load_state_dict(checkpoint['state_dict_c'])
        model_s.load_state_dict(checkpoint['state_dict_s'])
        if args.use_ema:
            ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
        optimizer_c.load_state_dict(checkpoint['optimizer_c'])
        scheduler_c.load_state_dict(checkpoint['scheduler_c'])
        optimizer_s.load_state_dict(checkpoint['optimizer_s'])
        scheduler_s.load_state_dict(checkpoint['scheduler_s'])

    # if args.amp:
    #     from apex import amp
    #     model, optimizer = amp.initialize(
    #         model, optimizer, opt_level=args.opt_level)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)


    model_c.zero_grad()
    model_s.zero_grad()
    if not args.eval_only:
        logger.info("***** Running training *****")
        logger.info(f"  Task = {args.dataset}@{args.num_labeled}")
        logger.info(f"  Num Epochs = {args.epochs}")
        logger.info(f"  Batch size per GPU = {args.batch_size}")
        logger.info(f"  Total train batch size = {args.batch_size*args.world_size}")
        logger.info(f"  Total optimization steps = {args.total_steps}")
        train(args, labeled_trainloader, unlabeled_dataset, test_loader, val_loader, ood_loaders,
              (model_c, model_s), (optimizer_c, optimizer_s), ema_model, (scheduler_c, scheduler_s))
    else:
        logger.info("***** Running Evaluation *****")
        logger.info(f"  Task = {args.dataset}@{args.num_labeled}")
        eval_model(args, labeled_trainloader, unlabeled_dataset, test_loader, val_loader,
             model_c, ema_model)


if __name__ == '__main__':
    main()
