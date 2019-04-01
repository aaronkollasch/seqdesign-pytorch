#!/usr/bin/env python
import numpy as np
import torch
import torch.utils.data as data

import data_loaders
import autoregressive_model
import autoregressive_train
import model_logging


def _init_fn(worker_id):
    np.random.seed(12 + worker_id)


USE_CUDA = True
device = torch.device("cuda:0" if USE_CUDA and torch.cuda.is_available() else "cpu")

model = autoregressive_model.AutoregressiveFR()
model.to(device)
dataset = data_loaders.DoubleWeightedNanobodyDataset(
    batch_size=48,
    dataset='datasets/nanobodies/Manglik_filt_seq_id80_id90.fa', working_dir='.',
    matching=True, unlimited_epoch=True,
)
loader = data.DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    collate_fn=dataset.collate_fn,
    pin_memory=True,
    worker_init_fn=_init_fn
)
trainer = autoregressive_train.AutoregressiveTrainer(
    model=model,
    data_loader=loader,
    snapshot_path='./snapshots',
    snapshot_name='nanobodies',
    device=device,
    logger=model_logging.TensorboardLogger()
)
print("Num parameters:", model.parameter_count())

trainer.train()
