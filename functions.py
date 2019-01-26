import math

import torch
import torch.nn.functional as F
import torch.autograd as autograd


def gelu(x):
    """BERT's implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT_TO_FUN = {
    'elu': F.elu,
    'relu': F.relu,
    'lrelu': F.leaky_relu,
    'gelu': gelu,
    'swish': swish,
    'none': lambda x: x,
}


def nonlinearity(nonlin_type):
    return ACT_TO_FUN[nonlin_type]


def comb_losses(losses_f, losses_r):
    losses_comb = {}
    for key in losses_f.keys():
        if 'per_seq' in key:
            losses_comb[key] = torch.stack([losses_f[key], losses_r[key]])
        else:
            losses_comb[key] = losses_f[key] + losses_r[key]
            losses_comb[key + '_f'] = losses_f[key]
            losses_comb[key + '_r'] = losses_r[key]
    return losses_comb


def l2_normalize(w, dim, eps=1e-12):
    """PyTorch implementation of tf.nn.l2_normalize
    """
    return w / w.pow(2).sum(dim, keepdim=True).clamp(min=eps).sqrt()


def l2_norm_except_dim(w, dim, eps=1e-12):
    norm_dims = [i for i, _ in enumerate(w.shape)]
    del norm_dims[dim]
    return l2_normalize(w, norm_dims, eps)


def moments(x, dim, keepdim=False):
    """PyTorch implementation of tf.nn.moments over a single dimension
    """
    # n = x.numel() / torch.prod(torch.tensor(x.shape)[dim])  # useful for multiple dims
    mean = x.mean(dim=dim, keepdim=True)
    variance = (x - mean.detach()).pow(2).mean(dim=dim, keepdim=keepdim)
    if not keepdim:
        mean = mean.squeeze(dim)
    return mean, variance


class Normalize(autograd.Function):
    """Normalize x across dim
    """
    @staticmethod
    def forward(ctx, x, dim, eps=1e-5):
        x_mu = x - x.mean(dim=dim, keepdim=True)
        inv_std = 1 / (x_mu.pow(2).mean(dim=dim, keepdim=True) + eps).sqrt()
        x_norm = x_mu * inv_std

        if ctx is not None:
            ctx.save_for_backward(x_mu, inv_std)
            ctx.dim = dim
        return x_norm

    @staticmethod
    def backward(ctx, grad_out):
        x_mu, inv_std = ctx.saved_tensors
        dim = ctx.dim
        n = x_mu.size(dim)

        # adapted from: https://cthorey.github.io/backpropagation/
        #               https://wiseodd.github.io/techblog/2016/07/04/batchnorm/
        dx = inv_std / n * (
                 grad_out * n -
                 grad_out.sum(dim, keepdim=True) -
                 (grad_out * x_mu).sum(dim, keepdim=True) * x_mu * inv_std ** 2
             )
        return dx, None, None

    @staticmethod
    def test():
        x = torch.DoubleTensor(3, 4, 2, 5).normal_(0, 1).requires_grad_()
        inputs = (x, 1)
        return autograd.gradcheck(Normalize.apply, inputs)


normalize = Normalize.apply
