import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dist
import torch.nn.functional as F

import layers
from utils import recursive_update


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


class Autoregressive(nn.Module):
    """An autoregressive model

    """
    model_type = 'autoregressive'

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

        self.step = 0
        self.image_summaries = {}

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
        self.image_summaries['LayerFeatures'] = dict(img=up_val_1d.permute(0, 1, 3, 2).detach(), max_outputs=3)
        if enc_params['dropout_type'] == "final":
            up_val_1d = self.final_dropout(up_val_1d)
        up_val_1d = self.end_conv(up_val_1d)
        return up_val_1d

    @staticmethod
    def reconstruction_loss(seq_logits, target_seqs, mask):
        seq_reconstruct = F.log_softmax(seq_logits, 1)
        # cross_entropy = F.cross_entropy(seq_logits, target_seqs.argmax(1), reduction='none')
        cross_entropy = F.nll_loss(seq_reconstruct, target_seqs.argmax(1), reduction='none')
        cross_entropy = cross_entropy * mask.squeeze(1)
        reconstruction_per_seq = cross_entropy.sum([1, 2])
        bitperchar_per_seq = reconstruction_per_seq / mask.sum([1, 2, 3])
        reconstruction_loss = reconstruction_per_seq.mean()
        bitperchar = bitperchar_per_seq.mean()
        return {
            'seq_reconstruct': seq_reconstruct,
            'ce_loss': reconstruction_loss,
            'ce_loss_per_seq': reconstruction_per_seq,
            'bitperchar': bitperchar,
            'bitperchar_per_seq': bitperchar_per_seq
        }

    def calculate_loss(
            self, seq_logits, target_seqs, mask, n_eff
    ):
        """

        :param seq_logits: (N, C, 1, L)
        :param target_seqs: (N, C, 1, L) as one-hot
        :param mask: (N, 1, 1, L)
        :param n_eff:
        :return:
        """
        hyperparams = self.hyperparams

        # cross-entropy
        reconstruction_loss = self.reconstruction_loss(
            seq_logits, target_seqs, mask
        )
        loss_per_seq = reconstruction_loss['ce_loss_per_seq']
        loss = reconstruction_loss['ce_loss']

        if hyperparams["optimization"]["l2_regularization"] or hyperparams["optimization"]["bayesian"]:

            # regularization
            weight_cost = torch.stack(self.weight_costs()).sum() / n_eff
            weight_cost = weight_cost * hyperparams["optimization"]["l2_lambda"]
            kl_weight_loss = weight_cost
            kl_loss = weight_cost

            # merge losses
            loss = loss + weight_cost

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
                loss = loss + self._anneal_embedding(self.step) * kl_logits_loss
            else:
                kl_embedding_loss = kl_weight_loss

        else:
            weight_cost = None
            kl_embedding_loss = torch.zeros([])
            kl_loss = torch.zeros([])

        seq_reconstruct = reconstruction_loss.pop('seq_reconstruct')
        self.image_summaries['SeqReconstruct'] = dict(img=seq_reconstruct.permute(0, 1, 3, 2).detach(), max_outputs=3)
        self.image_summaries['SeqTarget'] = dict(img=target_seqs.permute(0, 1, 3, 2).detach(), max_outputs=3)
        self.image_summaries['SeqDelta'] = dict(
            img=(seq_reconstruct - target_seqs).permute(0, 1, 3, 2).detach(), max_outputs=3)

        output = {
            'loss': loss,
            'ce_loss': None,
            'bitperchar': None,
            'loss_per_seq': loss_per_seq,
            'bitperchar_per_seq': None,
            'ce_loss_per_seq': None,
            'kl_embedding_loss': kl_embedding_loss,
            'kl_loss': kl_loss,
            'weight_cost': weight_cost,
        }
        output.update(reconstruction_loss)
        return output


class AutoregressiveFR(nn.Module):
    sub_model_class = Autoregressive
    model_type = 'autoregressive_fr'

    def __init__(
            self,
            **kwargs
    ):
        super(AutoregressiveFR, self).__init__()
        self.model = nn.ModuleDict()
        self.model['model_f'] = self.sub_model_class(**kwargs)
        self.model['model_r'] = self.sub_model_class(**kwargs)
        self.dims = self.model['model_f'].dims
        self.hyperparams = self.model['model_f'].hyperparams

        # make dictionaries the same
        self.model['model_r'].dims = self.model['model_f'].dims
        self.model['model_r'].hyperparams = self.model['model_f'].hyperparams

    @property
    def step(self):
        return self.model['model_f'].step

    @step.setter
    def step(self, new_step):
        self.model['model_f'].step = new_step
        self.model['model_r'].step = new_step

    @property
    def image_summaries(self):
        img_summaries_f = self.model['model_f'].image_summaries
        img_summaries_r = self.model['model_r'].image_summaries
        img_summaries = {}
        for key in img_summaries_f.keys():
            img_summaries[key + '_f'] = img_summaries_f[key]
            img_summaries[key + '_r'] = img_summaries_r[key]
        return img_summaries

    def weight_costs(self):
        return tuple(cost for model in self.model.children() for cost in model.weight_costs())

    def parameter_count(self):
        return sum(model.parameter_count() for model in self.model.children())

    def forward(self, input_f, mask_f, input_r, mask_r):
        output_logits_f = self.model['model_f'](input_f, mask_f)
        output_logits_r = self.model['model_r'](input_r, mask_r)
        return output_logits_f, output_logits_r

    def reconstruction_loss(
            self,
            seq_logits_f, target_seqs_f, mask_f,
            seq_logits_r, target_seqs_r, mask_r,
    ):
        losses_f = self.model['model_f'].reconstruction_loss(
            seq_logits_f, target_seqs_f, mask_f
        )
        losses_r = self.model['model_r'].reconstruction_loss(
            seq_logits_r, target_seqs_r, mask_r
        )
        return comb_losses(losses_f, losses_r)

    def calculate_loss(
            self,
            seq_logits_f, target_seqs_f, mask_f, n_eff_f,
            seq_logits_r, target_seqs_r, mask_r, n_eff_r
    ):
        losses_f = self.model['model_f'].calculate_loss(
            seq_logits_f, target_seqs_f, mask_f, n_eff_f
        )
        losses_r = self.model['model_r'].calculate_loss(
            seq_logits_r, target_seqs_r, mask_r, n_eff_r
        )
        return comb_losses(losses_f, losses_r)


class AutoregressiveVAE(nn.Module):
    """An autoregressive variational autoencoder

    """
    model_type = 'autoregressive_vae'

    def __init__(
            self,
            dims=None,
            hyperparams=None,
            channels=None,
            r_seed=None,
            dropout_p=None,
    ):
        super(AutoregressiveVAE, self).__init__()

        self.dims = {
            "batch": 10,
            "alphabet": 21,
            "length": 256,
            "embedding_size": 1
        }
        if dims is not None:
            self.dims.update(dims)

        self.hyperparams = {
            "encoder": {
                "channels": 48,
                "nonlinearity": "elu",
                "embedding_nnet_nonlinearity": "elu",
                "num_dilation_blocks": 3,
                "num_layers": 9,
                "dilation_schedule": None,
                "transformer": False,
                "inverse_temperature": False,
                "embedding_nnet_size": 200,
                "latent_size": 30,
                "dropout_p": 0.3,
                "config": "original",
            },
            "decoder": {
                "channels": 48,
                "nonlinearity": "relu",
                "num_dilation_blocks": 1,
                "num_layers": 5,
                "dilation_schedule": None,
                "transformer": False,
                "inverse_temperature": False,
                "positional_embedding": True,
                "pos_emb_max_len": 400,
                "pos_emb_step": 5,
                "config": "original",
                "dropout_p": 0.5,
            },
            "sampler_hyperparams": {
                'warm_up': 10000,
                'annealing_type': 'linear',
                'anneal_kl': True,
                'anneal_noise': True
            },
            "embedding_hyperparams": {
                'warm_up': 10000,
                'annealing_type': 'piecewise_linear',
                'anneal_kl': True,
                'anneal_noise': True
            },
            "random_seed": 42,
            "optimization": {
                "l2_regularization": True,
                "bayesian": True,
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

        # initialize encoder
        enc_params = self.hyperparams['encoder']
        nonlin = self._nonlinearity(enc_params['nonlinearity'])

        self.encoder = nn.ModuleDict()
        self.encoder.start_conv = layers.Conv2d(
            self.dims['alphabet'],
            enc_params['channels'],
            kernel_width=(1, 1),
            activation=nonlin,
        )

        self.encoder.dilation_blocks = nn.ModuleList()
        for block in range(enc_params['num_dilation_blocks']):
            self.encoder.dilation_blocks.append(layers.ConvNet1D(
                channels=enc_params['channels'],
                layers=enc_params['num_layers'],
                dropout_p=enc_params['dropout_p'],
                causal=False,
                config=enc_params['config'],
                dilation_schedule=enc_params['dilation_schedule'],
                transpose=False,
                nonlinearity=nonlin,
            ))

        self.encoder.emb_mu_one = nn.Linear(enc_params['channels'], enc_params['embedding_nnet_size'])
        self.encoder.emb_log_sigma_one = nn.Linear(enc_params['channels'], enc_params['embedding_nnet_size'])
        self.encoder.emb_mu_out = nn.Linear(enc_params['embedding_nnet_size'], enc_params['latent_size'])
        self.encoder.emb_log_sigma_out = nn.Linear(enc_params['embedding_nnet_size'], enc_params['latent_size'])

        # initialize decoder
        dec_params = self.hyperparams['decoder']
        nonlin = self._nonlinearity(dec_params['nonlinearity'])

        if dec_params['positional_embedding']:
            max_len = dec_params['pos_emb_max_len']
            step = dec_params['pos_emb_step']
            rbf_locations = torch.arange(step, max_len+1, step, dtype=torch.float32)
            rbf_locations = rbf_locations.view(1, dec_params['pos_emb_max_len'] // dec_params['pos_emb_step'], 1, 1)
            self.register_buffer('rbf_locations', rbf_locations)
        else:
            self.register_buffer('rbf_locations', None)

        self.decoder = nn.ModuleDict()
        self.decoder.start_conv = layers.Conv2d(
            (
                self.dims['alphabet'] +
                dec_params['pos_emb_max_len'] // dec_params['pos_emb_step']
                if dec_params['positional_embedding'] else 0 +
                enc_params['latent_size']
            ),
            dec_params['channels'],
            kernel_width=(1, 1),
            activation=nonlin,
        )

        self.decoder.dilation_blocks = nn.ModuleList()
        for block in range(dec_params['num_dilation_blocks']):
            self.decoder.dilation_blocks.append(layers.ConvNet1D(
                channels=dec_params['channels'],
                layers=dec_params['num_layers'],
                dropout_p=dec_params['dropout_p'],
                causal=True,
                config=dec_params['config'],
                dilation_schedule=dec_params['dilation_schedule'],
                transpose=False,
                nonlinearity=nonlin,
            ))

        self.decoder.end_conv = layers.Conv2d(
            dec_params['channels'],
            self.dims['alphabet'],
            kernel_width=(1, 1),
            g_init=0.1,
            activation=None,
        )

        self.step = 0
        self.forward_state = {'kl_embedding': None}
        self.image_summaries = {}  # TODO generate and handle summary images
        self._enable_gradient = 'ed'

    @property
    def enable_gradient(self):
        return self._enable_gradient

    @enable_gradient.setter
    def enable_gradient(self, value):
        if self._enable_gradient == value:
            return
        self._enable_gradient = value
        for p in self.encoder.parameters():
            p.requires_grad = 'e' in value
            if 'e' not in value:
                # p.grad = None
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()
        for p in self.decoder.parameters():
            p.requires_grad = 'd' in value
            if 'd' not in value:
                # p.grad = None
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()

    @staticmethod
    def _nonlinearity(nonlin_type):
        if nonlin_type == "elu":
            return F.elu
        elif nonlin_type == "relu":
            return F.relu
        elif nonlin_type == "lrelu":
            return F.leaky_relu

    @staticmethod
    def _kl_standard_normal(mu, log_sigma):
        """ KL divergence between two Diagonal Gaussians """
        return -0.5 * (1.0 + 2.0 * log_sigma - mu.pow(2) - (2.0 * log_sigma).exp())

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

    def _anneal(self, step):
        warm_up = self.hyperparams["sampler_hyperparams"]["warm_up"]
        annealing_type = self.hyperparams["sampler_hyperparams"]["annealing_type"]
        if annealing_type == "linear":
            return np.minimum(step / warm_up, 1.)
        elif annealing_type == "piecewise_linear":
            return np.minimum(torch.sigmoid(torch.tensor(step-warm_up).float()).item() * ((step-warm_up)/warm_up), 1.)
        elif annealing_type == "sigmoid":
            slope = self.hyperparams["sampler_hyperparams"]["sigmoid_slope"]
            return torch.sigmoid(torch.tensor(slope * (step - warm_up))).item()

    def _anneal_embedding(self, step):
        warm_up = self.hyperparams["embedding_hyperparams"]["warm_up"]
        annealing_type = self.hyperparams["embedding_hyperparams"]["annealing_type"]
        if annealing_type == "linear":
            return np.minimum(step / warm_up, 1.)
        elif annealing_type == "piecewise_linear":
            return np.minimum(torch.sigmoid(torch.tensor(step-warm_up).float()).item() * ((step-warm_up)/warm_up), 1.)
        elif annealing_type == "sigmoid":
            slope = self.hyperparams["embedding_hyperparams"]["sigmoid_slope"]
            return torch.sigmoid(torch.tensor(slope * (step-warm_up))).item()

    def sampler(self, mu, log_sigma, stddev=1.):
        if self.hyperparams["embedding_hyperparams"]["anneal_noise"]:
            stddev = self._anneal_embedding(self.step)
        return dist.Normal(mu, log_sigma.exp() * stddev).rsample()

    def weight_costs(self):
        return (
            self.decoder.start_conv.weight_costs() +
            tuple(cost for layer in self.decoder.dilation_blocks for cost in layer.weight_costs()) +
            self.decoder.end_conv.weight_costs()
        )

    def parameter_count(self):
        return sum(np.prod(param.shape) for param in self.parameters())

    def forward(self, inputs, input_masks):
        """
        :param inputs: (N, C_in, 1, L)
        :param input_masks: (N, 1, 1, L)
        :return:
        """
        # encoder
        enc_params = self.hyperparams['encoder']

        up_val_1d = self.encoder.start_conv(inputs)
        for convnet in self.encoder.dilation_blocks:
            up_val_1d = convnet(up_val_1d, input_masks)

        nonlin = self._nonlinearity(enc_params['embedding_nnet_nonlinearity'])
        up_val_1d = up_val_1d * input_masks
        up_val_mu_logsigma_2d = up_val_1d.sum(dim=[2, 3]) / input_masks.sum(dim=[2, 3])

        up_val_mu_2d = nonlin(self.encoder.emb_mu_one(up_val_mu_logsigma_2d))
        up_val_log_sigma_2d = nonlin(self.encoder.emb_log_sigma_one(up_val_mu_logsigma_2d))

        mu_2d = self.encoder.emb_mu_out(up_val_mu_2d)
        log_sigma_2d = self.encoder.emb_log_sigma_out(up_val_log_sigma_2d)

        self.forward_state['kl_embedding'] = self._kl_standard_normal(mu_2d, log_sigma_2d)
        z_2d = self.sampler(mu_2d, log_sigma_2d)

        self.image_summaries['mu'] = dict(
            img=mu_2d.unsqueeze(-1).unsqueeze(-1).permute(2, 1, 0, 3).detach(), max_outputs=1)
        self.image_summaries['log_sigma'] = dict(
            img=log_sigma_2d.unsqueeze(-1).unsqueeze(-1).permute(2, 1, 0, 3).detach(), max_outputs=1)
        self.image_summaries['z'] = dict(
                img=z_2d.unsqueeze(-1).unsqueeze(-1).permute(2, 1, 0, 3).detach(), max_outputs=1)

        # decoder
        dec_params = self.hyperparams['decoder']

        z = z_2d.unsqueeze(-1).unsqueeze(-1).expand((-1, -1, 1, inputs.size(3)))
        if dec_params['positional_embedding']:
            number_range = torch.arange(0, inputs.size(3), dtype=inputs.dtype, device=inputs.device)
            number_range = number_range.view(1, 1, 1, inputs.size(3))
            pos_embed = torch.exp(-0.5 * (number_range - self.rbf_locations).pow(2))
            pos_embed = pos_embed.expand((inputs.size(0), -1, 1, -1))
        else:
            pos_embed = torch.tensor([], dtype=inputs.dtype, device=inputs.device)
        input_1d = torch.cat([inputs, z, pos_embed], 1)

        up_val_1d = self.decoder.start_conv(input_1d)
        for convnet in self.decoder.dilation_blocks:
            up_val_1d = convnet(up_val_1d, input_masks)
        self.image_summaries['LayerFeatures'] = dict(img=up_val_1d.permute(0, 1, 3, 2).detach(), max_outputs=3)

        up_val_1d = self.decoder.end_conv(up_val_1d)

        return up_val_1d

    @staticmethod
    def reconstruction_loss(seq_logits, target_seqs, mask):
        """
        :param seq_logits: (N, C, 1, L)
        :param target_seqs: (N, C, 1, L) as one-hot
        :param mask: (N, 1, 1, L)
        """
        seq_reconstruct = F.log_softmax(seq_logits, 1)
        cross_entropy = F.nll_loss(seq_reconstruct, target_seqs.argmax(1), reduction='none')
        cross_entropy = cross_entropy * mask.squeeze(1)
        reconstruction_per_seq = cross_entropy.sum([1, 2])
        bitperchar_per_seq = reconstruction_per_seq / mask.sum([1, 2, 3])
        reconstruction_loss = reconstruction_per_seq.mean()
        bitperchar = bitperchar_per_seq.mean()
        return {
            'seq_reconstruct': seq_reconstruct,
            'ce_loss': reconstruction_loss,
            'ce_loss_per_seq': reconstruction_per_seq,
            'bitperchar': bitperchar,
            'bitperchar_per_seq': bitperchar_per_seq
        }

    def calculate_loss(
            self, seq_logits, target_seqs, mask, n_eff
    ):
        """

        :param seq_logits: (N, C, 1, L)
        :param target_seqs: (N, C, 1, L) as one-hot
        :param mask: (N, 1, 1, L)
        :param n_eff:
        :return: dict
        """
        hyperparams = self.hyperparams

        # cross-entropy
        reconstruction_loss = self.reconstruction_loss(
            seq_logits, target_seqs, mask
        )

        # regularization
        weight_cost = torch.stack(self.weight_costs()).sum() / n_eff
        weight_cost = weight_cost * hyperparams["optimization"]["l2_lambda"]
        kl_weight_loss = weight_cost
        kl_loss = weight_cost

        # embedding calculation
        embed_cost_per_seq = self.forward_state['kl_embedding'].sum(1)
        kl_embedding_loss = embed_cost_per_seq.mean()

        # merge losses
        loss_per_seq = reconstruction_loss['ce_loss_per_seq']
        loss = reconstruction_loss['ce_loss'] + weight_cost + kl_embedding_loss * self._anneal_embedding(self.step)

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
            loss = loss + self._anneal_embedding(self.step) * kl_logits_loss

        seq_reconstruct = reconstruction_loss.pop('seq_reconstruct')
        self.image_summaries['SeqReconstruct'] = dict(img=seq_reconstruct.permute(0, 1, 3, 2).detach(), max_outputs=3)
        self.image_summaries['SeqTarget'] = dict(img=target_seqs.permute(0, 1, 3, 2).detach(), max_outputs=3)
        self.image_summaries['SeqDelta'] = dict(
            img=(seq_reconstruct - target_seqs).permute(0, 1, 3, 2).detach(), max_outputs=3)

        output = {
            'loss': loss,
            'ce_loss': None,
            'bitperchar': None,
            'loss_per_seq': loss_per_seq,
            'bitperchar_per_seq': None,
            'ce_loss_per_seq': None,
            'kl_embedding_loss': kl_embedding_loss,
            'kl_loss': kl_loss,
            'weight_cost': weight_cost,
        }
        output.update(reconstruction_loss)
        return output


class AutoregressiveVAEFR(AutoregressiveFR):
    sub_model_class = AutoregressiveVAE
    model_type = 'autoregressive_vae_fr'

    def __init__(
            self,
            **kwargs
    ):
        super(AutoregressiveVAEFR, self).__init__(**kwargs)

    @property
    def enable_gradient(self):
        return self.model.model_f.enable_gradient

    @enable_gradient.setter
    def enable_gradient(self, value):
        self.model.model_f.enable_gradient = value
        self.model.model_r.enable_gradient = value

    def encoder_parameters(self):
        return list(self.model.model_f.encoder.parameters()) + list(self.model.model_r.encoder.parameters())

    def decoder_parameters(self):
        return list(self.model.model_f.decoder.parameters()) + list(self.model.model_r.decoder.parameters())
