# code referenced from https://github.com/vincentherrmann/pytorch-wavenet/blob/master/model_logging.py
import threading
from io import BytesIO
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.tensorboard import SummaryWriter


class Accumulator:
    def __init__(self, *keys):
        self._values = {key: 0. for key in keys}
        self.log_interval = 0

    def accumulate(self, **kwargs):
        for key in kwargs:
            self._values[key] += kwargs[key]
        self.log_interval += 1

    def reset(self):
        for key in self._values:
            self._values[key] = 0.
        self.log_interval = 0

    @property
    def values(self):
        return {key: value / self.log_interval for key, value in self._values.items()}

    def __getattr__(self, item):
        return self._values[item] / self.log_interval


class Logger:
    def __init__(self,
                 log_interval=50,
                 validation_interval=200,
                 generate_interval=500,
                 info_interval=1000,
                 trainer=None,
                 generate_function=None):
        self.trainer = trainer
        self.log_interval = log_interval
        self.val_interval = validation_interval
        self.gen_interval = generate_interval
        self.info_interval = info_interval
        self.log_time = time.time()
        self.load_time = 0.
        self.accumulator = Accumulator('loss', 'ce_loss', 'bitperchar')
        self.generate_function = generate_function
        if self.generate_function is not None:
            self.generate_thread = threading.Thread(target=self.generate_function)
            self.generate_function.daemon = True

    def log(self, current_step, current_losses, current_grad_norm, load_time=0.):
        self.load_time += load_time
        self.accumulator.accumulate(
            loss=float(current_losses['loss'].detach()),
            ce_loss=float(current_losses['ce_loss'].detach()),
            bitperchar=float(current_losses['bitperchar'].detach()) if 'bitperchar' in current_losses else 0.,
        )

        if current_step % self.log_interval == 0 or current_step < 10:
            self.log_loss(current_step)
            self.log_time = time.time()
            self.load_time = 0.
            self.accumulator.reset()
        if self.val_interval is not None and self.val_interval > 0 and current_step % self.val_interval == 0:
            self.validate(current_step)
        if self.gen_interval is not None and self.gen_interval > 0 and current_step % self.gen_interval == 0:
            self.generate(current_step)
        if self.info_interval is not None and self.info_interval > 0 and current_step % self.info_interval == 0:
            self.info(current_step)

    def log_loss(self, current_step):
        v = self.accumulator.values
        print(f"{time.time() - self.log_time:7.3f} {self.load_time:7.3f} "
              f"loss, ce_loss, bitperchar at step {current_step:8d}: "
              f"{v['loss']:11.6f}, {v['ce_loss']:11.6f}, {v['bitperchar']:10.6f}", flush=True)

    def validate(self, current_step):
        avg_loss, avg_accuracy = self.trainer.validate()
        print("validation loss: " + str(avg_loss), flush=True)
        print("validation accuracy: " + str(avg_accuracy * 100) + "%", flush=True)

    def generate(self, current_step):
        if self.generate_function is None:
            return

        if self.generate_thread.is_alive():
            print("Last generate is still running, skipping this one")
        else:
            self.generate_thread = threading.Thread(target=self.generate_function, args=[current_step])
            self.generate_thread.daemon = True
            self.generate_thread.start()

    def info(self, current_step):
        pass
        # print(
        #     'GPU Mem Allocated:',
        #     round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1),
        #     'GB, ',
        #     'Cached:',
        #     round(torch.cuda.memory_cached(0) / 1024 ** 3, 1),
        #     'GB'
        # )


# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
class TensorboardLogger(Logger):
    def __init__(self,
                 log_interval=50,
                 validation_interval=200,
                 generate_interval=500,
                 info_interval=1000,
                 trainer=None,
                 generate_function=None,
                 log_dir='logs',
                 log_param_histograms=False,
                 log_image_summaries=True,
                 print_output=False,
                 ):
        super().__init__(
            log_interval, validation_interval, generate_interval, info_interval, trainer, generate_function)
        self.writer = SummaryWriter(log_dir)
        self.log_param_histograms = log_param_histograms
        self.log_image_summaries = log_image_summaries
        self.print_output = print_output

    def log(self, current_step, current_losses, current_grad_norm, load_time=0.):
        super(TensorboardLogger, self).log(current_step, current_losses, current_grad_norm, load_time)
        self.scalar_summary('grad norm', current_grad_norm, current_step)
        self.scalar_summary('loss', current_losses['loss'].detach(), current_step)
        self.scalar_summary('ce_loss', current_losses['ce_loss'].detach(), current_step)
        if 'accuracy' in current_losses:
            self.scalar_summary('accuracy', current_losses['accuracy'].detach(), current_step)
        if 'bitperchar' in current_losses:
            self.scalar_summary('bitperchar', current_losses['bitperchar'].detach(), current_step)
        self.scalar_summary('reconstruction loss', current_losses['ce_loss'].detach(), current_step)
        self.scalar_summary('regularization loss', current_losses['weight_cost'].detach(), current_step)

    def log_loss(self, current_step):
        if self.print_output:
            Logger.log_loss(self, current_step)
        # loss
        v = self.accumulator.values
        avg_loss, avg_ce_loss, avg_bitperchar = v['loss'], v['ce_loss'], v['bitperchar']
        self.scalar_summary('avg loss', avg_loss, current_step)
        self.scalar_summary('avg ce loss', avg_ce_loss, current_step)
        self.scalar_summary('avg bitperchar', avg_bitperchar, current_step)

        if self.log_param_histograms:
            for tag, value, in self.trainer.model.named_parameters():
                tag = tag.replace('.', '/')
                self.histo_summary(tag, value.data, current_step)
                if value.grad is not None:
                    self.histo_summary(tag + '/grad', value.grad.data, current_step)

        if self.log_image_summaries:
            for tag, summary in self.trainer.model.image_summaries.items():
                self.image_summary(tag, summary['img'], current_step, max_outputs=summary.get('max_outputs', 3))

    def validate(self, current_step):
        avg_loss, avg_accuracy = self.trainer.validate()
        self.scalar_summary('validation loss', avg_loss, current_step)
        self.scalar_summary('validation accuracy', avg_accuracy, current_step)
        if self.print_output:
            print("validation loss: " + str(avg_loss), flush=True)
            print("validation accuracy: " + str(avg_accuracy * 100) + "%", flush=True)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        if isinstance(value, torch.Tensor):
            value = value.item()  # value must have 1 element only
        self.writer.add_scalar(tag, value, global_step=step)

    def image_summary(self, tag, images, step, max_outputs=3):
        """Log a tensor image.
        :param tag: string summary name
        :param images: (N, H, W, C) or (N, H, W)
        :param step: current step
        :param max_outputs: max N images to save
        """

        images = images[:max_outputs]
        format = "NHW" if images.dim() == 3 else "NHWC"
        self.writer.add_images(tag, images, global_step=step, dataformats=format)

    def histo_summary(self, tag, values, step, bins=200):
        """Log a histogram of the tensor of values."""
        self.writer.add_histogram(tag, values, global_step=step, bins=bins)
