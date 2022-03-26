import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch import distributed as dist


# ------------------- SSL utils -----------------------
# Supervised learning and SSL
class TransformSLAndSSL:
    def __init__(self, orig_transform, transform_ssl):
        self.orig_transform = orig_transform
        self.transform_ssl = transform_ssl

    def __call__(self, inp):
        out = self.orig_transform(inp)
        out_ssl_1 = self.transform_ssl(inp)
        out_ssl_2 = self.transform_ssl(inp)
        return out, out_ssl_1, out_ssl_2

# (H,W): data input size
def get_simclr_transforms(H, W, orig_transform):
    simclr_aug = transforms.Compose([
        transforms.RandomResizedCrop((H,W)),
        transforms.RandomHorizontalFlip(),  # with 0.5 probability
        transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),])
    return TransformSLAndSSL(orig_transform, simclr_aug)
# -----------------------------------------------------

# transform a scalar to multihot vector
# targets: (N,)
def scalar2onehot(targets, num_class):
    N = targets.shape[0]
    onehot_target = torch.zeros(N, num_class).to(targets.device).scatter_(1, targets.unsqueeze(-1), 1)
    return onehot_target

def scalar2SmoothOneHot(targets, num_class):
    N = targets.shape[0]
    hot_prob = 0.9
    smooth_prob = (1.0 - hot_prob) / (N-1)
    onehot_target = (smooth_prob*torch.ones(N, num_class)).to(targets.device).scatter_(1, targets.unsqueeze(-1), hot_prob)
    return onehot_target

# Cutmix
# assuming the targets are one-hot vectors
def cut_and_mix(images, targets):
    N,_,H,W = images.shape
    assert H==W, 'only support square right now'

    cut_len = int(np.floor(np.random.rand()*H))
    cut_len = max(min(cut_len, H-1), 1)
    mix_ratio = float(cut_len*cut_len) / (H*W)

    top = np.random.randint(0, H - cut_len)
    left = np.random.randint(0, W - cut_len)
    bottom = top + cut_len
    right = left + cut_len

    # cut and mix
    # shuffled batch images
    rp = torch.randperm(N)
    shuffled_images = images[rp,:,:,:]
    shuffled_targets = targets[rp,:]
    images[:, :, top:bottom, left:right] = shuffled_images[:, :, top:bottom, left:right]

    # adjust the target
    targets = (1-mix_ratio)*targets + mix_ratio*shuffled_targets

    return images, targets

# Cutmix w/ small window
# assuming the targets are one-hot vectors
def cut_and_mix_small_window(images, targets):
    N,_,H,W = images.shape
    assert H==W, 'only support square right now'

    cut_len = int(np.floor(0.4*np.random.rand()*H))
    cut_len = max(min(cut_len, H-1), 1)
    mix_ratio = float(cut_len*cut_len) / (H*W)

    top = np.random.randint(0, H - cut_len)
    left = np.random.randint(0, W - cut_len)
    bottom = top + cut_len
    right = left + cut_len

    # cut and mix
    # shuffled batch images
    rp = torch.randperm(N)
    shuffled_images = images[rp,:,:,:]
    # shuffled_targets = targets[rp,:]
    images[:, :, top:bottom, left:right] = shuffled_images[:, :, top:bottom, left:right]

    # do not adjust the target
    # targets = (1-mix_ratio)*targets + mix_ratio*shuffled_targets

    return images, targets

# Mixup
# assuming the targets are one-hot vectors
def mixup(images, targets):
    lam = np.random.beta(1.0, 1.0)
    N = images.shape[0]
    rp = torch.randperm(N)
    shuffled_images = images[rp,:,:,:]
    shuffled_targets = targets[rp,:]   

    images = lam * images + (1-lam)*shuffled_images
    targets = lam * targets + (1-lam)*shuffled_targets

    return images, targets

class MultiLabelCrossEntropyLoss(nn.Module):

    def __init__(self):
        super(MultiLabelCrossEntropyLoss, self).__init__()

    # logit: (N, C)
    # label: (N, C)
    def forward(self, logits, label):
        loss = -(label*F.log_softmax(logits, dim=1)).sum(dim=-1).mean()
        return loss

class BalancedCrossEntropy(nn.Module):

    def __init__(self, tao):
        super(BalancedCrossEntropy, self).__init__()
        self.tao = tao
        self.eps = 1e-8

    def forward(self, logits, label):
        num_classes = logits.shape[1]
        label_onehot = scalar2onehot(label, num_classes)

        # v1 (undesired solution)
        # loss = -self.tao*(logits*label).sum(dim=-1) + (2.0 - self.tao)*torch.log(self.eps + logits.exp().sum(dim=-1))

        # v2 (undesired solution)
        loss = -self.tao*(logits*label_onehot).sum(dim=-1) + torch.log(self.eps + logits.exp().sum(dim=-1))

        # v3 (broken solution)
        # pos_loss = -self.tao*(logits*label).sum(dim=-1)
        # neg_loss = (2.0 - self.tao)*torch.log(self.eps + ((1.-label)*logits.exp()).sum(dim=-1))
        # loss = pos_loss + neg_loss

        # v4 (broken solution)
        # pos_loss = -self.tao*(logits*label).sum(dim=-1)
        # reweight = 1. - (1. - self.tao)*label
        # neg_loss = torch.log(self.eps + (reweight*logits).exp().sum(dim=-1))
        # loss = pos_loss + neg_loss
        # return loss.mean()

        # v5
        # reweight = 1. - (1. - self.tao)*label_onehot
        # logits = reweight*logits
        # loss = nn.CrossEntropyLoss(None)(logits, label)
        # return loss

        # pos_loss = -(logits*label_onehot).sum(dim=-1)
        # reweight = (1. - label_onehot)*self.tao + label_onehot
        # neg_loss = torch.log(self.eps + (reweight*logits.exp()).sum(dim=-1))
        # loss = pos_loss + neg_loss

        return loss.mean()

class AugCrossEntropy(nn.Module):

    def __init__(self, n_aug):
        super(AugCrossEntropy, self).__init__()
        self.n_aug = n_aug
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, features, targets):
        N,C = features.shape
        device = features.device
        # first generating random fake class embeddings
        pseudo_embeddings = F.normalize(torch.rand(self.n_aug, C, device=device) - 0.5, dim=-1)
        pseudo_logits = torch.matmul(features, pseudo_embeddings.t())
        cat_logits = torch.cat([logits, pseudo_logits], dim=-1)
        loss = self.ce(cat_logits, targets)
        return loss


# -------------- DDP utils -------------------
def reduce_tensor_mean(tensor, n):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n
    return rt

def reduce_tensor_sum(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt

def global_gather(x):
    all_x = [torch.ones_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(all_x, x, async_op=False)
    return torch.cat(all_x, dim=0)

# differentiable gather layer
class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out
# --------------------------------------------

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
