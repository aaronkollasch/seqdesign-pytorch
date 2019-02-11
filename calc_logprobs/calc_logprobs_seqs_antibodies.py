#!/usr/bin/env python
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.append("..")
import autoregressive_model
import autoregressive_train
import data_loaders
from tf_reader import TFReader

parser = argparse.ArgumentParser(description="Calculate the log probability of mutated sequences.")
parser.add_argument("--restore", type=str, default='', required=True,
                    help="Snapshot name for restoring the model")
parser.add_argument("--input", type=str, default='', required=True,
                    help="Directory and filename of the input data.")
parser.add_argument("--output", type=str, default='output', required=True,
                    help="Directory and filename of the output data.")
parser.add_argument("--channels", type=int, default=48,
                    help="Number of channels.")
parser.add_argument("--num-samples", type=int, default=1,
                    help="Number of iterations to run the model.")
parser.add_argument("--batch-size", type=int, default=100,
                    help="Minibatch size for inferring effect prediction.")
parser.add_argument("--include-vh", action='store_true',
                    help="Include an encoding of the VH gene in the input. "
                         "Overridden by checkpoint parameter if set.")
parser.add_argument("--dropout-p", type=float, default=0.,
                    help="Dropout p while sampling log p(x) (drop rate, not keep rate)")
parser.add_argument("--num-data-workers", type=int, default=0,
                    help="Number of workers to load data")
parser.add_argument("--no-cuda", action='store_true',
                    help="Disable GPU evaluation")
parser.add_argument("--from-tf", action='store_true',
                    help="Load model from tensorflow checkpoint")

args = parser.parse_args()

print("OS: ", sys.platform)
print("Python: ", sys.version)
print("PyTorch: ", torch.__version__)
print("Numpy: ", np.__version__)

USE_CUDA = not args.no_cuda
device = torch.device("cuda:0" if USE_CUDA and torch.cuda.is_available() else "cpu")
print('Using device:', device)
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3, 1), 'GB')
print()

dataset = data_loaders.VHAntibodyFastaDataset(
    batch_size=args.batch_size,
    working_dir='.',
    dataset=args.input,
    matching=True,
    unlimited_epoch=False,
    include_vh=args.include_vh,
)
loader = data_loaders.GeneratorDataLoader(
    dataset,
    num_workers=args.num_data_workers,
    pin_memory=True,
)
print("Read in test data")

print("Initializing and loading variables")
if args.from_tf:
    dims = {'input': dataset.input_dim}
    hyperparams = {'encoder': {'dilation_schedule': [1, 2, 4, 8, 16, 32, 64, 128, 200],
                               "config": "original",
                               'dropout_type': 'independent'}}
    model = autoregressive_model.AutoregressiveFR(
        channels=args.channels, dropout_p=args.dropout_p, dims=dims, hyperparams=hyperparams)
    reader = TFReader(os.path.join('../snapshots', args.restore))
    reader.load_autoregressive_fr(model)
else:
    checkpoint = torch.load(os.path.join('../snapshots', args.restore), map_location='cpu')
    dims = checkpoint['model_dims']
    hyperparams = checkpoint['model_hyperparams']
    model = autoregressive_model.AutoregressiveFR(dims=dims, hyperparams=hyperparams, dropout_p=args.dropout_p)
    model.load_state_dict(checkpoint['model_state_dict'])
    if 'dataset_params' in checkpoint:
        dataset.params = checkpoint['dataset_params']
model.to(device)
print("Num parameters:", model.parameter_count())

trainer = autoregressive_train.AutoregressiveTrainer(
    model=model,
    data_loader=None,
    device=device,
)
output = trainer.test(loader, model_eval=False, num_samples=args.num_samples)
output = pd.DataFrame(output, columns=output.keys())
output.to_csv(args.output, index=False)
print("Done!")
