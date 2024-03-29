import torch
import torch.nn.functional as F
from bypass_bn import disable_running_stats, enable_running_stats

def cal_stats(out):
    mu = torch.mean(out, dim=0)
    logvar = torch.log(torch.var(out, dim=0))
    return mu, logvar

def compute_rec(reconstructed_inputs, original_inputs):
    return F.mse_loss(reconstructed_inputs, original_inputs, reduction="mean")


def cal_entropy(out):
    # entropy
    return - ((out.softmax(1) * F.log_softmax(out, 1)).sum(1)).mean()


def cal_vae_loss(mu, logvar):
    return - 0.01 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def cal_content_classification(content_pred, class_label, mask=None):
    if mask:
        content_cls_loss = (F.cross_entropy(content_pred, class_label, reduction="none", ignore_index=-1) * mask.float()).mean()
    else:
        content_cls_loss = F.cross_entropy(content_pred, class_label)
    
    return content_cls_loss


def cal_style_classification(style_pred, aug_label):
    style_cls_loss = F.cross_entropy(style_pred, aug_label)

    return style_cls_loss


def cal_reconstruction(args, rec_content, rec_style, rec_target):
    content_rec_loss = args.rec_coef * (compute_rec(rec_content, rec_target))
    style_rec_loss = args.rec_coef * (compute_rec(rec_style, rec_target))

    return content_rec_loss, style_rec_loss


    
def cal_ssl(args, logits_all, mask_probs):
    
    logits_w, logits_s = logits_all.chunk(2)
    pseudo_label = torch.softmax(logits_w.detach()/args.T, dim=-1)
    max_probs, max_target = torch.max(pseudo_label, dim=-1)
    fix_mask = max_probs.ge(args.threshold).float()
    ssl_consistency_loss = (F.cross_entropy(logits_s, max_target, reduction='none') * fix_mask).mean()
    mask_probs.update(fix_mask.mean().item())

    return ssl_consistency_loss



def cal_ood(args, logits_open_all, targets_l, b_size=None, negative=False):
    logits_open_all = logits_open_all.view(logits_open_all.size(0), 2, -1)
    logits_open_all = F.softmax(logits_open_all, 1)
    
    if not negative:
        open_target = torch.zeros((targets_l.size(0), logits_open_all.size(2))).to(targets_l.device)
        label_range = torch.arange(0, targets_l.size(0)).long()
        open_target[label_range, targets_l] = 1
        open_target_nega = 1 - open_target

        if logits_open_all.size(0) > 2*b_size:
            logits_open_l = logits_open_all[:2*b_size]
            logits_open_u_w1, logits_open_u_w2 = logits_open_all[2*b_size:].chunk(2)
            # ood consistency
            ood_loss = args.lambda_socr * torch.mean(torch.sum(torch.sum(torch.abs(
                logits_open_u_w1 - logits_open_u_w2)**2, 1), 1))

            # entropy minimization
            ood_loss += 0.5 * args.lambda_oem * (torch.mean(torch.mean(torch.sum(-logits_open_u_w1 * torch.log(logits_open_u_w1 + 1e-8), 1), 1)) + \
                            torch.mean(torch.mean(torch.sum(-logits_open_u_w2 * torch.log(logits_open_u_w2 + 1e-8), 1), 1)))

            # open classification
            ood_loss += torch.mean(torch.sum(-torch.log(logits_open_l[:, 1, :] + 1e-8) * open_target, 1) + torch.max(-torch.log(logits_open_l[:, 0, :] + 1e-8) * open_target_nega, 1)[0])
        else:
            # open classification
            ood_loss = torch.mean(torch.sum(-torch.log(logits_open_all[:, 1, :] + 1e-8) * open_target, 1) + torch.max(-torch.log(logits_open_all[:, 0, :] + 1e-8) * open_target_nega, 1)[0])

    else:
        open_target = torch.ones((targets_l.size(0), logits_open_all.size(2))).to(targets_l.device)
        # negative detection
        ood_loss = torch.mean(torch.max(-torch.log(logits_open_all[:, 0, :] + 1e-8) * open_target, 1)[0])

    return ood_loss



def augmentation(args, mean, std, model_c, model_s, inputs, class_label, domain_label=None, aug_type='benign', domain_targeted=False):
    """
        This function computs the adversarial examples by augmenting content and style.
    """ 
    mean = torch.tensor(mean).view(3,1,1)
    std = torch.tensor(std).view(3,1,1)
    upper_limit = ((1 - mean) / std).to(args.device)
    lower_limit = ((0 - mean) / std).to(args.device)
    
    cls_target = torch.zeros((class_label.size(0), args.num_classes)).to(class_label.device)
    label_range = torch.arange(0, class_label.size(0)).long()
    cls_target[label_range, class_label] = 1
    cls_target_nega = 1 - cls_target
    if not domain_targeted:
        domain_label = torch.randint(args.aug_num, (class_label.size(0),)).long().to(class_label.device)
    
    disable_running_stats(model_c)
    disable_running_stats(model_s)
    model_c.eval()
    model_s.eval()

    perturbations = torch.zeros_like(inputs)
    perturbations.uniform_(-0.01, 0.01)
    perturbations.data.clamp_(lower_limit - inputs, upper_limit - inputs)
    perturbations.requires_grad = True

    with torch.no_grad():
        init_cls, _, init_content, _ = model_c(inputs)
        init_cls_hat = model_s.disentangle(init_content)
        init_dom, _, init_style, _ = model_s(inputs)
        
    content_losses = 0
    style_losses = 0
    # adv augmentation
    for i in range(args.adv_step):
        if perturbations.grad is not None:
            perturbations.grad.data.zero_()

        adv_inputs = inputs + perturbations

        # intervine content
        cls_pred, _, adv_content, _ = model_c(adv_inputs)
        if aug_type == 'malign':
            # change content
            cls_pred = cls_pred.softmax(dim=1)
            content_loss = torch.mean(torch.max(-torch.log(cls_pred + 1e-8) * cls_target_nega, 1)[0])
        else:
            # maintain content
            content_loss =  F.mse_loss(adv_content, init_content)
        content_losses += content_loss.item()
        content_loss.backward()

        grad = perturbations.grad.data
        grad_norm = grad / (grad.reshape(grad.size()[0], -1).norm(dim=1)[:, None, None, None] + 1e-8)
        perturbations.data = (perturbations + grad_norm * args.adv_magnitude).clamp_(lower_limit - inputs, upper_limit - inputs)
        del grad, grad_norm, content_loss, cls_pred, adv_content

        if perturbations.grad is not None:
            perturbations.grad.data.zero_()
        adv_inputs = inputs + perturbations
        # intervine style
        dom_pred, _, adv_style, _  = model_s(adv_inputs)
        if aug_type == 'malign':
            # maintain style
            style_loss = F.mse_loss(adv_style, init_style)
        else:
            # change style with new domain label
            style_loss  = F.cross_entropy(dom_pred, domain_label)
        style_losses += style_loss.item()
        style_loss.backward()

        grad = perturbations.grad.data
        grad_norm = grad / (grad.reshape(grad.size()[0], -1).norm(dim=1)[:, None, None, None] + 1e-8)
        perturbations.data = (perturbations + grad_norm * args.adv_magnitude).clamp_(lower_limit - inputs, upper_limit - inputs)
        del grad, grad_norm, style_loss, dom_pred, adv_style

    adv_inputs = inputs + perturbations

    enable_running_stats(model_c)
    enable_running_stats(model_s)
    model_c.train()
    model_s.train()
    return adv_inputs.detach(), content_losses, style_losses




