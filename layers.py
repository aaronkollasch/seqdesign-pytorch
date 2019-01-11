import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.autograd as autograd


class HyperparameterError(Exception):
    pass


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
    eps = 1e-5

    @staticmethod
    def forward(ctx, x, dim):
        x_mu = x - x.mean(dim=dim, keepdim=True)
        var = x_mu.pow(2).mean(dim=dim, keepdim=True)
        inv_std = 1 / (var + Normalize.eps).sqrt()
        del var

        if ctx is not None:
            ctx.save_for_backward(x_mu, inv_std)
            ctx.dim = dim

        x_norm = x_mu * inv_std
        return x_norm

    @staticmethod
    def backward(ctx, grad_out):
        x_mu, inv_std = ctx.saved_tensors
        dim = ctx.dim
        n = x_mu.size(dim)

        # # derived from https://wiseodd.github.io/techblog/2016/07/04/batchnorm/
        # go = grad_out * inv_std
        # dvar = (grad_out * x_mu).sum(dim, keepdim=True) * -0.5 * inv_std ** 3
        # dmu = - go.sum(dim, keepdim=True) + 2 * dvar * x_mu.mean(dim, keepdim=True)
        # dx = go + 2 * dvar * x_mu / n + dmu / n

        # simplified as in http://cthorey.github.io/backpropagation/
        dx = inv_std / n * (
                 grad_out * n -
                 grad_out.sum(dim, keepdim=True) -
                 (grad_out * x_mu).sum(dim, keepdim=True) * x_mu * inv_std ** 2
             )

        return dx, None

    @staticmethod
    def test():
        x = torch.DoubleTensor(3, 4, 2, 5).normal_(0, 1).requires_grad_()
        inputs = (x, 1)
        test = autograd.gradcheck(Normalize.apply, inputs)
        return test

    @staticmethod
    def test2():
        x_in = torch.DoubleTensor(3, 4, 1, 5).normal_(0, 1).requires_grad_()
        inputs = (x_in, 1)

        Normalize.forward(None, *inputs).pow(2).mean().backward()
        x1 = x_in.grad.clone()
        x_in.grad.data.zero_()
        Normalize.apply(*inputs).pow(2).mean().backward()
        x2 = x_in.grad.clone()
        # print(x2 / x1)
        return torch.allclose(x1, x2)


normalize = Normalize.apply


class LayerChannelNorm(nn.Module):
    """Normalizes across C for input NCHW
    """
    __constants__ = ['num_channels', 'dim', 'eps', 'affine', 'weight', 'bias', 'g_init', 'bias_init']

    def __init__(self, num_channels, dim=1, eps=1e-5, affine=True, g_init=1.0, bias_init=0.1):
        super(LayerChannelNorm, self).__init__()
        self.num_channels = num_channels
        self.dim = dim
        self.eps = eps
        self.affine = affine
        self.g_init = g_init
        self.bias_init = bias_init
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_channels))
            self.bias = Parameter(torch.Tensor(num_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            nn.init.normal_(self.weight, mean=self.g_init, std=1e-6)
            nn.init.normal_(self.bias, mean=self.bias_init, std=1e-6)

    def forward(self, x):
        # mean, var = moments(x, self.dim, keepdim=True)
        # h = (x - mean) / (var + self.eps).sqrt()
        h = normalize(x, self.dim)

        if self.affine:
            shape = [1 for _ in x.shape]
            shape[self.dim] = self.num_channels
            h = self.weight.view(shape) * h + self.bias.view(shape)
        return h


class LayerNorm(nn.GroupNorm):
    """Normalizes across CHW for input NCHW
    """
    def __init__(self, num_channels, eps=1e-5, affine=True, g_init=1.0, bias_init=0.1):
        self.g_init = g_init
        self.bias_init = bias_init
        super(LayerNorm, self).__init__(num_groups=1, num_channels=num_channels, eps=eps, affine=affine)

    def reset_parameters(self):
        if self.affine:
            nn.init.normal_(self.weight, mean=self.g_init, std=1e-6)
            nn.init.normal_(self.bias, mean=self.bias_init, std=1e-6)


class Conv2d(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_width=(1, 1),
            stride=(1, 1),
            dilation=(1, 1),
            g_init=1.0,
            bias_init=0.1,
            causal=False,
            activation=None,
    ):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_width = kernel_width
        self.stride = stride
        self.dilation = dilation
        self.causal = causal
        self.activation = activation

        padding = []
        for w, d in zip(kernel_width, dilation):
            padding.append(d * (w-1)//2)
        self.padding = tuple(padding)

        self.bias = Parameter(torch.Tensor(out_channels))
        self.weight_v = Parameter(torch.Tensor(out_channels, in_channels, *kernel_width))
        self.weight_g = Parameter(torch.Tensor(out_channels))

        if causal:
            if any(w % 2 == 0 for w in kernel_width):
                raise HyperparameterError(f"Even kernel width incompatible with causal convolution: {kernel_width}")
            if kernel_width == (1, 3):  # make common case explicit
                mask = torch.Tensor([1., 1., 0.])
            elif kernel_width[0] == 1:
                mask = torch.ones(kernel_width)
                mask[0, kernel_width[1] // 2 + 1:] = 0
            else:
                mask = torch.ones(kernel_width)
                mask[kernel_width[0] // 2, kernel_width[1] // 2:] = 0
                mask[kernel_width[0] // 2 + 1:, :] = 0

            mask = mask.view(1, 1, *kernel_width)
            self.register_buffer('mask', mask)
        else:
            self.register_buffer('mask', None)

        self.reset_parameters(g_init=g_init, bias_init=bias_init)

    def reset_parameters(self, v_mean=0., v_std=0.05, g_init=1.0, bias_init=0.1):
        nn.init.normal_(self.weight_v, mean=v_mean, std=v_std)
        nn.init.constant_(self.weight_g, val=g_init)
        nn.init.constant_(self.bias, val=bias_init)

    def weight_costs(self):
        return (
            self.weight_v.pow(2).sum(),
            self.weight_g.pow(2).sum(),
            self.bias.pow(2).sum()
        )

    def forward(self, inputs):
        """
        :param inputs: (N, C_in, H, W)
        :return: (N, C_out, H, W)
        """
        shape = (self.out_channels, 1, 1, 1)
        weight = l2_norm_except_dim(self.weight_v, 0) * self.weight_g.view(shape)
        if self.mask is not None:
            weight = weight * self.mask

        h = F.conv2d(inputs, weight, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation)
        if self.activation is not None:
            h = self.activation(h)
        return h

    def extra_repr(self):
        s = '{in_channels}, {out_channels}, kernel_size={kernel_width}'
        if self.stride != (1,) * len(self.stride):
            s += ', stride={stride}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.causal:
            s += ', causal=True'
        return s.format(**self.__dict__)


class ConvNet1DLayer(nn.Module):
    configurations = ['original', 'standard']

    def __init__(
            self,
            channels=48,
            dilation=1,
            dropout_p=0.5,
            causal=True,
            config='original',
            add_input_channels=None,
            transpose=False,
            nonlinearity=F.relu,
    ):
        super(ConvNet1DLayer, self).__init__()
        self.channels = channels
        self.dilation = dilation
        self.causal = causal
        self.dropout_p = dropout_p
        self.add_input_channels = None if add_input_channels == 0 else add_input_channels
        self.transpose = transpose
        self.nonlinearity = nonlinearity
        self.config = config

        if config not in self.configurations:
            raise HyperparameterError(f"Unknown configuration: '{config}'. Accepts {self.configurations}")

        if self.add_input_channels is not None:
            input_channels = channels + self.add_input_channels
        else:
            input_channels = channels

        self.layernorm_1 = LayerChannelNorm(input_channels)
        self.layernorm_2 = LayerChannelNorm(channels)
        if config == 'standard':
            self.layernorm_3 = LayerChannelNorm(channels)
        else:
            self.register_parameter('layernorm_3', None)

        self.mix_conv_1 = Conv2d(input_channels, channels)
        self.dilated_conv = Conv2d(
            channels, channels,
            kernel_width=(1, 3),
            dilation=(1, dilation),
            causal=causal,
            bias_init=0.0,
        )
        self.mix_conv_3 = Conv2d(channels, channels)

        self.dropout = nn.Dropout(p=dropout_p)

    def weight_costs(self):
        return (
            self.mix_conv_1.weight_costs() +
            self.dilated_conv.weight_costs() +
            self.mix_conv_3.weight_costs()
        )

    def forward(self, inputs, input_masks, additional_input=None):
        """
        :param inputs: Tensor(N, C, 1, L)
        :param input_masks: Tensor(N, 1, 1, L)
        :param additional_input: Tensor(N, C_add, 1, L)
        :return: Tensor(N, C, 1, L)
        """
        if self.add_input_channels is not None:
            delta_layer = torch.cat([inputs, self.additional_input], dim=1)
        else:
            delta_layer = inputs

        if self.config == 'original':
            delta_layer = self.layernorm_1(delta_layer)
            delta_layer = self.mix_conv_1(delta_layer)
            delta_layer = self.nonlinearity(delta_layer)

            delta_layer = self.dilated_conv(delta_layer)
            delta_layer = self.nonlinearity(delta_layer)

            delta_layer = self.mix_conv_3(delta_layer)
            delta_layer = self.nonlinearity(delta_layer)

            delta_layer = self.dropout(delta_layer)
            delta_layer = self.layernorm_2(delta_layer)

        elif self.config == 'standard':
            delta_layer = self.layernorm_1(delta_layer)
            delta_layer = self.nonlinearity(delta_layer)
            delta_layer = self.mix_conv_1(delta_layer)

            delta_layer = self.layernorm_2(delta_layer)
            delta_layer = self.nonlinearity(delta_layer)
            delta_layer = self.dilated_conv(delta_layer)

            delta_layer = self.layernorm_3(delta_layer)
            delta_layer = self.nonlinearity(delta_layer)
            delta_layer = self.mix_conv_3(delta_layer)

            delta_layer = self.dropout(delta_layer)

        return delta_layer

    def extra_repr(self):
        return '{channels}, dilation={dilation}, causal={causal}, config={config}, ' \
               'add_input_channels={add_input_channels}'.format(**self.__dict__)


class ConvNet1D(nn.Module):
    additional_input_layers = ['all', 'first']

    def __init__(
            self,
            channels=48,
            layers=9,
            dropout_p=0.5,
            causal=True,
            config='original',  # 'original', 'standard'
            add_input_channels=None,
            add_input_layer=None,  # 'all', 'first'
            dilation_schedule=None,
            transpose=False,
            nonlinearity=F.elu,
    ):
        super(ConvNet1D, self).__init__()
        self.channels = channels
        self.num_layers = layers
        self.causal = causal
        self.dropout_p = dropout_p
        self.transpose = transpose
        self.nonlinearity = nonlinearity
        self.add_input_channels = add_input_channels
        self.add_input_layer = add_input_layer
        self.config = config

        if add_input_layer is not None and add_input_layer not in self.additional_input_layers:
            raise HyperparameterError(f"Unknown additional input layer: '{add_input_layer}'. "
                                      f"Accepts {self.additional_input_layers}")

        if dilation_schedule is None:
            self.dilations = [2 ** i for i in range(layers)]
        else:
            self.dilations = dilation_schedule

        self.dilation_layers = nn.ModuleList()

        for i_layer, dilation in enumerate(self.dilations):
            add_input_c = None
            if self.add_input_layer == 'all' or (self.add_input_layer == 'first' and i_layer == 0):
                add_input_c = add_input_channels

            self.dilation_layers.append(ConvNet1DLayer(
                channels=channels, dilation=dilation, dropout_p=dropout_p, causal=causal, config=config,
                add_input_channels=add_input_c, transpose=transpose, nonlinearity=nonlinearity
            ))

        if causal:
            self.receptive_field = 2 ** (layers-1)
        else:
            self.receptive_field = 2 ** layers - 1

    def weight_costs(self):
        return [cost for layer in self.dilation_layers for cost in layer.weight_costs()]

    def forward(self, inputs, input_masks, additional_input=None):
        """
        :param inputs: Tensor(N, C, 1, L)
        :param input_masks: Tensor(N, 1, 1, L)
        :param additional_input: Tensor(N, C_add, 1, L)
        :return: Tensor(N, C, 1, L)
        """
        up_layer = inputs

        for layer, dilation in enumerate(self.dilations):
            add_input = None
            if self.add_input_layer == 'all' or (self.add_input_layer == 'first' and layer == 0):
                add_input = additional_input

            delta_layer = self.dilation_layers[layer](up_layer, input_masks, add_input)
            up_layer = up_layer + delta_layer

        return up_layer

    def extra_repr(self):
        return '{channels}, layers={num_layers}, causal={causal}, config={config}, ' \
               'add_input_channels={add_input_channels}'.format(**self.__dict__)
