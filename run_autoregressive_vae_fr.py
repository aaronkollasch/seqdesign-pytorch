#!/usr/bin/env python
import sys
import argparse
import time

import numpy as np
import torch

import data_loaders
import autoregressive_model
import autoregressive_train
import model_logging

working_dir = '/n/groups/marks/users/aaron/autoregressive'
data_dir = '/n/groups/marks/projects/autoregressive'

parser = argparse.ArgumentParser(description="Train an autoregressive model on a collection of sequences.")
parser.add_argument("--channels", type=int, default=48,
                    help="Number of channels.")
parser.add_argument("--num-iterations", type=int, default=250005,
                    help="Number of iterations to run the model.")
parser.add_argument("--dataset", type=str, default=None,
                    help="Dataset name for fitting model. Alignment weights must be computed beforehand.")
parser.add_argument("--num-data-workers", type=int, default=4,
                    help="Number of workers to load data")
parser.add_argument("--restore", type=str, default=None,
                    help="Snapshot path for restoring a model to continue training.")
parser.add_argument("--r-seed", type=int, default=42,
                    help="Random seed")
parser.add_argument("--no-lag-inf", action='store_true',
                    help="Disable lagging inference")
parser.add_argument("--no-cuda", action='store_true',
                    help="Disable GPU training")
args = parser.parse_args()

run_name = f"{args.dataset}_elu_channels-{args.channels}_rseed-{args.r_seed}" \
    f"_start-{time.strftime('%y%b%d_%H%M', time.localtime())}"

sbatch_executable = f"""#!/bin/bash
#SBATCH -c 4                               # Request one core
#SBATCH -N 1                               # Request one node (if you request more than one core with -c, also using
                                           # -N 1 means all cores will be on the same node)
#SBATCH -t 2-11:59                         # Runtime in D-HH:MM format
#SBATCH -p gpu                             # Partition to run in
#SBATCH --gres=gpu:1
#SBATCH --mem=30G                          # Memory total in MB (for all cores)
#SBATCH -o slurm_files/slurm-%j.out        # File to which STDOUT + STDERR will be written, including job ID in filename
hostname
pwd
module load gcc/6.2.0 cuda/9.0
srun stdbuf -oL -eL {sys.executable} \\
  {sys.argv[0]} \\
  --dataset {args.dataset} --num-iterations {args.num_iterations} --channels {args.channels} --r_seed {args.r_seed} \\
  --restore {{restore}}
"""

torch.manual_seed(args.r_seed)
torch.cuda.manual_seed_all(args.r_seed)


def _init_fn(worker_id):
    np.random.seed(args.r_seed + worker_id)


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

print("Run:", run_name)

# Variables for runtime modification
batch_size = 30
num_iterations = args.num_iterations

plot_train = 100
dropout_p_train = 0.5

# fitness_check = 2
# fitness_start = 2
# num_iterations = 36

dataset = data_loaders.SingleFamilyDataset(
    batch_size=batch_size,
    working_dir=data_dir,
    dataset=args.dataset,
    matching=True,
    unlimited_epoch=True,
)
loader = data_loaders.GeneratorDataLoader(
    dataset,
    num_workers=args.num_data_workers,
    pin_memory=True,
    worker_init_fn=_init_fn
)

model = autoregressive_model.AutoregressiveVAEFR(channels=args.channels, dropout_p=dropout_p_train)
model.to(device)

trainer = autoregressive_train.AutoregressiveVAETrainer(
    model=model,
    data_loader=loader,
    snapshot_path=working_dir + '/snapshots',
    snapshot_name=run_name,
    snapshot_interval=args.num_iterations // 10,
    snapshot_exec_template=sbatch_executable,
    device=device,
    # logger=model_logging.Logger(validation_interval=None),
    logger=model_logging.TensorboardLogger(
        log_interval=500,
        validation_interval=1000,
        generate_interval=5000,
        log_dir=working_dir + '/logs/' + run_name
    )
)
if args.restore is not None:
    print("Restoring model from:", args.restore)
    trainer.load_state(args.restore)
if args.no_lag_inf:
    trainer.params['lagging_inference'] = False
print("Num parameters:", model.parameter_count())

trainer.train(steps=num_iterations)
