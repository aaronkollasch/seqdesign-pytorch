import collections

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dist
import torch.nn.functional as F

import layers


def recursive_update(orig_dict, update_dict):
    for key, val in update_dict.items():
        if isinstance(val, collections.Mapping):
            orig_dict[key] = recursive_update(orig_dict.get(key, {}), val)
        else:
            orig_dict[key] = val
    return orig_dict


class Autoregressive(nn.Module):
    """An autoregressive model

    """
    def __init__(
            self,
            dims=None,
            hyperparams=None,
            channels=None,
            r_seed=None,
            dropout_p=None,
    ):
        super(Autoregressive, self).__init__()

        self.dims = {
            "batch": 10,
            "alphabet": 21,
            "length": 256,
            "embedding_size": 1
        }
        if dims is not None:
            self.dims.update(dims)

        self.hyperparams = {
            # For purely dilated conv network
            "encoder": {
                "channels": 48,
                "nonlinearity": "elu",
                "num_dilation_blocks": 6,
                "num_layers": 9,
                "dilation_schedule": None,
                "transformer": False,  # TODO transformer
                "inverse_temperature": False,
                "dropout_type": "inter",  # options = "final", "inter", "gaussian"  # TODO what is gaussian?
                "dropout_p": 0.5,  # probability of zeroing out value
                "config": "original",  # options = "original", "standard"
            },
            "sampler_hyperparams": {
                'warm_up': 1,
                'annealing_type': 'linear',
                'anneal_kl': True,
                'anneal_noise': True
            },
            "embedding_hyperparams": {
                'warm_up': 1,
                'annealing_type': 'linear',
                'anneal_kl': True,
                'anneal_noise': False
            },
            "random_seed": 42,
            "optimization": {
                "l2_regularization": True,
                "bayesian": False,
                "l2_lambda": 1.,
                "bayesian_logits": False,
                "mle_logits": False,
            }
        }
        if hyperparams is not None:
            recursive_update(self.hyperparams, hyperparams)
        if channels is not None:
            self.hyperparams['encoder']['channels'] = channels
        if dropout_p is not None:
            self.hyperparams['encoder']['dropout_p'] = dropout_p
        if r_seed is not None:
            self.hyperparams['random_seed'] = r_seed

        self.model_type = 'autoregressive'

        # build encoder
        enc_params = self.hyperparams['encoder']
        nonlin = self._nonlinearity(enc_params['nonlinearity'])

        self.start_conv = layers.Conv2d(
            self.dims['alphabet'],
            enc_params['channels'],
            kernel_width=(1, 1),
            activation=None,
        )

        self.dilation_blocks = nn.ModuleList()
        for block in range(enc_params['num_dilation_blocks']):
            self.dilation_blocks.append(layers.ConvNet1D(
                channels=enc_params['channels'],
                layers=enc_params['num_layers'],
                dropout_p=enc_params['dropout_p'],
                causal=True,
                config=enc_params['config'],
                dilation_schedule=enc_params['dilation_schedule'],
                transpose=False,
                nonlinearity=nonlin,
            ))

        if enc_params['dropout_type'] == "final":
            self.final_dropout = nn.Dropout(max(enc_params['dropout_p']-0.3, 0.))
        else:
            self.register_parameter('final_dropout', None)

        self.end_conv = layers.Conv2d(
            enc_params['channels'],
            self.dims['alphabet'],
            kernel_width=(1, 1),
            g_init=0.1,
            activation=None,
        )

    @staticmethod
    def _nonlinearity(nonlin_type):
        if nonlin_type == "elu":
            return F.elu
        elif nonlin_type == "relu":
            return F.relu
        elif nonlin_type == "lrelu":
            return F.leaky_relu

    @staticmethod
    def _log_gaussian(z, prior_mu, prior_sigma):
        prior = dist.Normal(prior_mu, prior_sigma)
        return prior.log_prob(z)

    def _kl_mixture_gaussians(
            self, z, log_sigma, p=0.1,
            mu_one=0., mu_two=0., sigma_one=1., sigma_two=1.
    ):
        gauss_one = self._log_gaussian(z, mu_one, sigma_one)
        gauss_two = self._log_gaussian(z, mu_two, sigma_two)
        entropy = 0.5 * torch.log(2.0 * np.pi * np.e * torch.exp(2. * log_sigma))
        return (p * gauss_one) + ((1. - p) * gauss_two) + entropy

    def _mle_mixture_gaussians(
            self, z, p=0.1,
            mu_one=0., mu_two=0., sigma_one=1., sigma_two=1.
    ):
        gauss_one = self._log_gaussian(z, mu_one, sigma_one)
        gauss_two = self._log_gaussian(z, mu_two, sigma_two)
        return (p * gauss_one) + ((1. - p) * gauss_two)

    def weight_costs(self):
        return (
                self.start_conv.weight_costs() +
                tuple(cost for layer in self.dilation_blocks for cost in layer.weight_costs()) +
                self.end_conv.weight_costs()
        )

    def parameter_count(self):
        return sum(np.prod(param.shape) for param in self.parameters())

    def forward(self, inputs, input_masks):
        """
        :param inputs: (N, C_in, 1, L)
        :param input_masks: (N, 1, 1, L)
        :return:
        """
        enc_params = self.hyperparams['encoder']

        up_val_1d = self.start_conv(inputs)
        for convnet in self.dilation_blocks:
            up_val_1d = convnet(up_val_1d, input_masks)
        if enc_params['dropout_type'] == "final":
            up_val_1d = self.final_dropout(up_val_1d)
        up_val_1d = self.end_conv(up_val_1d)
        return up_val_1d

    def calculate_loss(
            self, seq_logits, target_seqs, mask, n_eff, step=0,
    ):
        """

        :param seq_logits: (N, C, 1, L)
        :param target_seqs: (N, C, 1, L) as one-hot
        :param mask: (N, 1, 1, L)
        :param step:
        :param n_eff:
        :return:
        """
        hyperparams = self.hyperparams

        # cross-entropy
        cross_entropy = F.cross_entropy(seq_logits, target_seqs.argmax(1), reduction='none')
        cross_entropy = cross_entropy * mask.squeeze(1)
        reconstruction_per_seq = cross_entropy.sum([1, 2])
        bitperchar_per_seq = reconstruction_per_seq / mask.sum([1, 2, 3])
        reconstruction_loss = reconstruction_per_seq.mean()
        bitperchar = bitperchar_per_seq.mean()

        if hyperparams["optimization"]["l2_regularization"] or hyperparams["optimization"]["bayesian"]:

            # regularization
            weight_cost = torch.stack(self.weight_costs()).sum() / n_eff
            weight_cost = weight_cost * hyperparams["optimization"]["l2_lambda"]
            kl_weight_loss = weight_cost
            kl_loss = weight_cost

            # merge losses
            loss_per_seq = reconstruction_per_seq
            loss = reconstruction_loss + weight_cost

            # KL loss
            if hyperparams["optimization"]["bayesian_logits"] or hyperparams["optimization"]["mle_logits"]:
                if self.hyperparams["optimization"]["mle_logits"]:
                    kl_logits = - self._mle_mixture_gaussians(
                        seq_logits, p=.6, mu_one=0., mu_two=0., sigma_one=1.25, sigma_two=3.
                    )
                else:
                    kl_logits = None

                kl_logits = kl_logits * mask
                kl_logits_per_seq = kl_logits.sum([1, 2, 3])
                loss_per_seq = loss_per_seq + kl_logits_per_seq
                kl_logits_loss = kl_logits_per_seq.mean()
                kl_loss += kl_logits_loss
                kl_embedding_loss = kl_logits_loss
                loss = loss + self._anneal_embedding(step) * kl_logits_loss
            else:
                kl_embedding_loss = kl_weight_loss

        else:
            weight_cost = None

            kl_embedding_loss = torch.zeros([])
            kl_loss = torch.zeros([])

            loss_per_seq = reconstruction_per_seq
            loss = reconstruction_loss

        return {
            'loss': loss,
            'ce_loss': reconstruction_loss,
            'bitperchar': bitperchar,
            'loss_per_seq': loss_per_seq,
            'bitperchar_per_seq': bitperchar_per_seq,
            'ce_loss_per_seq': reconstruction_per_seq,
            'kl_embedding_loss': kl_embedding_loss,
            'kl_loss': kl_loss,
            'weight_cost': weight_cost,
        }


class AutoregressiveFR(nn.Module):
    def __init__(
            self,
            **kwargs
    ):
        super(AutoregressiveFR, self).__init__()
        self.model_type = 'autoregressive_fr'
        self.model = nn.ModuleDict()
        self.model['model_f'] = Autoregressive(**kwargs)
        self.model['model_r'] = Autoregressive(**kwargs)
        self.dims = self.model['model_f'].dims
        self.hyperparams = self.model['model_f'].hyperparams

    def weight_costs(self):
        return tuple(cost for model in self.model.children() for cost in model.weight_costs())

    def parameter_count(self):
        return sum(model.parameter_count() for model in self.model.children())

    def forward(self, input_f, mask_f, input_r, mask_r):
        output_logits_f = self.model['model_f'](input_f, mask_f)
        output_logits_r = self.model['model_r'](input_r, mask_r)
        return output_logits_f, output_logits_r

    def calculate_loss(
            self,
            seq_logits_f, target_seqs_f, mask_f, n_eff_f,
            seq_logits_r, target_seqs_r, mask_r, n_eff_r,
            step=0,
    ):
        losses_f = self.model['model_f'].calculate_loss(
            seq_logits_f, target_seqs_f, mask_f, n_eff_f, step
        )
        losses_r = self.model['model_r'].calculate_loss(
            seq_logits_r, target_seqs_r, mask_r, n_eff_r, step
        )

        losses_comb = {}
        for key in losses_f.keys():
            if 'per_seq' in key:
                losses_comb[key] = torch.stack([losses_f[key], losses_r[key]])
            else:
                losses_comb[key] = losses_f[key] + losses_r[key]
                losses_comb[key + '_f'] = losses_f[key]
                losses_comb[key + '_r'] = losses_r[key]

        return losses_comb
