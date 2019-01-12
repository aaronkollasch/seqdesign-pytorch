import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data

from model_logging import Logger


class AutoregressiveTrainer:
    def __init__(
            self,
            model,
            data_loader,
            optimizer=optim.Adam,
            params=None,
            lr=None,
            weight_decay=None,
            gradient_clipping=None,
            logger=Logger(),
            snapshot_path=None,
            snapshot_name=None,
            snapshot_interval=None,
            snapshot_exec_template=None,
            device=torch.device('cpu')
    ):
        self.params = {
            'lr': 0.001,
            'weight_decay': 0,
            'clip': 100.0,
            'snapshot_path': None,
            'snapshot_name': 'snapshot',
            'snapshot_interval': 1000,
        }
        if params is not None:
            self.params.update(params)
        if lr is not None:
            self.params['lr'] = lr
        if weight_decay is not None:
            self.params['weight_decay'] = weight_decay
        if gradient_clipping is not None:
            self.params['clip'] = gradient_clipping
        if snapshot_path is not None:
            self.params['snapshot_path'] = snapshot_path
        if snapshot_name is not None:
            self.params['snapshot_name'] = snapshot_name
        if snapshot_interval is not None:
            self.params['snapshot_interval'] = snapshot_interval
        if snapshot_exec_template is not None:
            self.params['snapshot_exec_template'] = snapshot_exec_template

        self.model = model
        self.loader = data_loader

        self.run_fr = 'fr' in model.model_type
        self.optimizer_type = optimizer
        self.logger = logger
        self.logger.trainer = self
        self.current_step = 0
        self.device = device

        self.optimizer = self.optimizer_type(
            params=self.model.parameters(),
            lr=self.params['lr'], weight_decay=self.params['weight_decay'])

    def train(self, steps=1e8, continue_at_step=None):
        self.model.train()
        device = self.device

        data_iter = iter(self.loader)
        n_eff = self.loader.dataset.n_eff

        if continue_at_step is not None:
            self.current_step = continue_at_step

        print('    step  step-t load-t   loss       CE-loss    bitperchar   l2-norm', flush=True)
        for step in range(int(self.current_step), int(steps)):
            start = time.time()

            batch = next(data_iter)
            for key in batch.keys():
                batch[key] = batch[key].to(device, non_blocking=True)
            data_load_time = time.time()-start

            if self.run_fr:
                output_logits_f, output_logits_r = self.model(
                    batch['prot_decoder_input'], batch['prot_mask_decoder'],
                    batch['prot_decoder_input_r'], batch['prot_mask_decoder'])
                losses = self.model.calculate_loss(
                    output_logits_f, batch['prot_decoder_output'], batch['prot_mask_decoder'], n_eff,
                    output_logits_r, batch['prot_decoder_output_r'], batch['prot_mask_decoder'], n_eff, step)
            else:
                output_logits_f = self.model(batch['prot_decoder_input'], batch['prot_mask_decoder'])
                losses = self.model.calculate_loss(
                    output_logits_f, batch['prot_decoder_output'], batch['prot_mask_decoder'], n_eff, step)

            loss = losses['loss']
            ce_loss = losses['ce_loss']
            kl_loss = losses['kl_embedding_loss']
            bitperchar = losses['bitperchar']

            self.optimizer.zero_grad()
            loss.backward()

            if self.params['clip'] is not None:
                total_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.params['clip'])
            else:
                total_norm = 0.0

            nan_check = False
            if nan_check and (
                    any(torch.isnan(param).any() for param in self.model.parameters()) or
                    any(torch.isnan(param.grad).any() for param in self.model.parameters())
            ):
                print("nan detected:")
                print("{: 8d} {:6.3f} {:5.4f} {:11.6f} {:11.6f} {:11.8f} {:10.6f}".format(
                    step, time.time() - start, data_load_time,
                    loss.detach(), ce_loss.detach(), bitperchar.detach(), kl_loss.detach()))
                print('grad norm', total_norm)
                print('params', [name for name, param in self.model.named_parameters() if torch.isnan(param).any()])
                print('grads', [name for name, param in self.model.named_parameters() if torch.isnan(param.grad).any()])
                self.save_state(last_batch=batch)
                break

            self.optimizer.step()
            step += 1
            self.current_step = step

            if step % self.params['snapshot_interval'] == 0:
                if self.params['snapshot_path'] is None:
                    continue
                self.save_state()

            self.logger.log(step, losses, total_norm)
            print("{: 8d} {:6.3f} {:5.4f} {:11.6f} {:11.6f} {:11.8f} {:10.6f}".format(
                step, time.time()-start, data_load_time,
                loss.detach(), ce_loss.detach(), bitperchar.detach(), kl_loss.detach()), flush=True)

    def validate(self, batch_size=48):
        return 0.0, 0.0
        self.model.eval()
        with torch.no_grad():
            (
                prot_decoder_input_f, prot_decoder_output_f, prot_mask_decoder,
                prot_decoder_input_r, prot_decoder_output_r,
                n_eff
            ) = self.loader.dataset.generate_test_data(self, batch_size, matching=True)  # TODO write generate_test_data
            if self.run_fr:
                output_logits_f, output_logits_r = self.model(
                    prot_decoder_input_f, prot_mask_decoder, prot_decoder_input_r, prot_mask_decoder)
                output_logits = torch.cat([output_logits_f, output_logits_r], dim=0)
                target_seqs = torch.cat([prot_decoder_output_f, prot_decoder_output_r], dim=0)
                mask = torch.cat([prot_mask_decoder, prot_mask_decoder], dim=0)
            else:
                output_logits = self.model(prot_decoder_input_f, prot_mask_decoder)
                target_seqs = prot_decoder_output_f
                mask = prot_mask_decoder

            cross_entropy = F.cross_entropy(output_logits, target_seqs.argmax(1), reduction='none')
            cross_entropy = cross_entropy * mask.squeeze(1)
            reconstruction_per_seq = cross_entropy.sum([1, 2]) / mask.sum([1, 2, 3])
            reconstruction_loss = reconstruction_per_seq.mean()
            accuracy_per_seq = target_seqs[output_logits.argmax(1, keepdim=True)].sum([1, 2]) / mask.sum([1, 2, 3])
            avg_accuracy = accuracy_per_seq.mean()
        self.model.train()
        return reconstruction_loss, avg_accuracy

    def test(self, data_loader, model_eval=True, num_samples=1):
        if model_eval:
            self.model.eval()

        n_eff = data_loader.dataset.n_eff

        print('    step  step-t  CE-loss     bit-per-char', flush=True)
        for i_iter in range(num_samples):  # TODO implement sampling
            output = {
                'name': [],
                'mean': [],
                'bitperchar': [],
                'forward': [],
                'reverse': [],
                'sequence': []
            }
            if not self.run_fr:
                del output['forward']
                del output['reverse']

            for i_batch, batch in enumerate(data_loader):
                start = time.time()
                for key in batch.keys():
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device, non_blocking=True)

                with torch.no_grad():
                    if self.run_fr:
                        output_logits_f, output_logits_r = self.model(
                            batch['prot_decoder_input'], batch['prot_mask_decoder'],
                            batch['prot_decoder_input_r'], batch['prot_mask_decoder'])
                        losses = self.model.calculate_loss(
                            output_logits_f, batch['prot_decoder_output'], batch['prot_mask_decoder'], n_eff,
                            output_logits_r, batch['prot_decoder_output_r'], batch['prot_mask_decoder'], n_eff)
                    else:
                        output_logits_f = self.model(batch['prot_decoder_input'], batch['prot_mask_decoder'])
                        losses = self.model.calculate_loss(
                            output_logits_f, batch['prot_decoder_output'], batch['prot_mask_decoder'], n_eff)

                    ce_loss = losses['ce_loss_per_seq']
                    if self.run_fr:
                        ce_loss_mean = ce_loss.mean(0)
                    else:
                        ce_loss_mean = ce_loss
                    ce_loss_per_char = ce_loss_mean / batch['prot_mask_decoder'].sum([1, 2, 3])

                output['name'].extend(batch['names'])
                output['sequence'].extend(batch['sequences'])
                output['mean'].extend(ce_loss_mean.numpy())
                output['bitperchar'].extend(ce_loss_per_char.numpy())

                if self.run_fr:
                    ce_loss_f = ce_loss[0]
                    ce_loss_r = ce_loss[1]
                    output['forward'].extend(ce_loss_f.numpy())
                    output['reverse'].extend(ce_loss_r.numpy())

                print("{: 8d} {:6.3f} {:11.6f} {:11.6f}".format(
                    i_batch, time.time()-start, ce_loss_mean.mean(), ce_loss_per_char.mean()),
                    flush=True)

        self.model.train()
        return output

    def save_state(self, last_batch=None):
        snapshot = f"{self.params['snapshot_path']}/{self.params['snapshot_name']}_{self.current_step}.pth"
        revive_exec = f"{self.params['snapshot_path']}/revive_executable/{self.params['snapshot_name']}.sh"
        torch.save(
            {
                'step': self.current_step,
                'model_type': self.model.model_type,
                'model_state_dict': self.model.state_dict(),
                'model_dims': self.model.dims,
                'model_hyperparams': self.model.hyperparams,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_params': self.params,
                'last_batch': last_batch
            },
            snapshot
        )
        with open(revive_exec, "w") as f:
            snapshot_exec = self.params['snapshot_exec_template'].format(
                restore=os.path.abspath(snapshot)
            )
            f.write(snapshot_exec)

    def load_state(self, f_path, map_location=None):
        checkpoint = torch.load(f_path, map_location=map_location)
        if self.model.model_type != checkpoint['model_type']:
            print("Warning: model type mismatch: loaded type {} for model type {}".format(
                checkpoint['model_type'], self.model.model_type
            ))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_step = checkpoint['step']
        self.params.update(checkpoint['train_params'])
