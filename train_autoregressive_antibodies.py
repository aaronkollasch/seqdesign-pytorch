#!/usr/bin/env python
import sys
import argparse
import time
import json

import numpy as np
import torch

import data_loaders
import autoregressive_model
import autoregressive_train
import model_logging
from utils import get_cuda_version, get_cudnn_version

working_dir = '.'  # '/n/groups/marks/users/aaron/autoregressive'
data_dir = '.'  # '/n/groups/marks/users/aaron/autoregressive'

###################
# PARSE ARGUMENTS #
###################

parser = argparse.ArgumentParser(description="Train an autoregressive model on a collection of sequences.")
parser.add_argument("--channels", type=int, default=48,
                    help="Number of channels.")
parser.add_argument("--batch-size", type=int, default=32,
                    help="Batch size.")
parser.add_argument("--num-iterations", type=int, default=250005,
                    help="Number of iterations to run the model.")
parser.add_argument("--dataset", type=str, default=None,
                    help="Dataset name for fitting model. Alignment weights must be computed beforehand.")
parser.add_argument("--dataset-type", type=str, default='double_clustered',
                    help="Level of clustering in dataset to sample from.")
parser.add_argument("--num-data-workers", type=int, default=4,
                    help="Number of workers to load data")
parser.add_argument("--restore", type=str, default=None,
                    help="Snapshot path for restoring a model to continue training.")
parser.add_argument("--run-name", type=str, default=None,
                    help="Name of run")
parser.add_argument("--r-seed", type=int, default=42,
                    help="Random seed")
parser.add_argument("--dropout-p", type=float, default=0.5,
                    help="Dropout probability (drop rate, not keep rate)")
parser.add_argument("--no-cuda", action='store_true',
                    help="Disable GPU training")
args = parser.parse_args()


########################
# MAKE RUN DESCRIPTORS #
########################

if args.run_name is None:
    args.run_name = f"{args.dataset.split('/')[-1].split('.')[0]}_elu_channels-{args.channels}" \
        f"_dropout-{args.dropout_p}_rseed-{args.r_seed}" \
        f"_start-{time.strftime('%y%b%d_%H%M', time.localtime())}"

restore_args = [
    f"--dataset {args.dataset}",
    f"--num-iterations {args.num_iterations}",
    f"--channels {args.channels}",
    f"--batch-size {args.batch_size}",
    f"--dropout-p {args.dropout_p}",
    f"--r-seed {args.r_seed}",
    f"--restore {{restore}}",
    f"--run-name {args.run_name}",
]
restore_args = " \\\n  ".join(restore_args)

sbatch_executable = f"""#!/bin/bash
#SBATCH -c 4                               # Request one core
#SBATCH -N 1                               # Request one node (if you request more than one core with -c, also using
                                           # -N 1 means all cores will be on the same node)
#SBATCH -t 2-11:59                         # Runtime in D-HH:MM format
#SBATCH -p gpu                             # Partition to run in
#SBATCH --gres=gpu:1
#SBATCH --mem=40G                          # Memory total in MB (for all cores)
#SBATCH -o slurm_files/slurm-%j.out        # File to which STDOUT + STDERR will be written, including job ID in filename
hostname
pwd
module load gcc/6.2.0 cuda/9.0
srun stdbuf -oL -eL {sys.executable} \\
  {sys.argv[0]} \\
  {restore_args}
"""


####################
# SET RANDOM SEEDS #
####################

if args.restore is not None:
    # prevent from repeating batches/seed when restoring at intermediate point
    # script is repeatable as long as restored at same step
    # assumes restore arg of *[_/][step].pth
    args.r_seed += int(args.restore.split('_')[-1].split('/')[-1].split('.')[0])
    args.r_seed = args.r_seed % (2 ** 32 - 1)  # limit of np.random.seed

np.random.seed(args.r_seed)
torch.manual_seed(args.r_seed)
torch.cuda.manual_seed_all(args.r_seed)


def _init_fn(worker_id):
    np.random.seed(args.r_seed + worker_id)


#####################
# PRINT SYSTEM INFO #
#####################


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
    print(get_cuda_version())
    print("CuDNN Version ", get_cudnn_version())
print()

print("Run:", args.run_name)


#############
# LOAD DATA #
#############

if args.dataset_type in ['double', 'double_clustered']:
    data_loader = data_loaders.DoubleClusteredSequenceDataset
elif args.dataset_type in ['single', 'single_clustered']:
    data_loader = data_loaders.SingleClusteredSequenceDataset
else:
    data_loader = data_loaders.FastaDataset

dataset = data_loader(
    batch_size=args.batch_size,
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


##############
# LOAD MODEL #
##############

if args.restore is not None:
    print("Restoring model from:", args.restore)
    checkpoint = torch.load(args.restore, map_location='cpu' if device.type == 'cpu' else None)
    dims = checkpoint['model_dims']
    hyperparams = checkpoint['model_hyperparams']
    trainer_params = checkpoint['train_params']
    model = autoregressive_model.AutoregressiveFR(dims=dims, hyperparams=hyperparams, dropout_p=args.dropout_p)
else:
    dims = {'input': len(dataset.alphabet)}
    hyperparams = {'random_seed': args.r_seed}
    checkpoint = args.restore
    trainer_params = None
    model = autoregressive_model.AutoregressiveFR(
        dims=dims, hyperparams=hyperparams, channels=args.channels, dropout_p=args.dropout_p)
model.to(device)


################
# RUN TRAINING #
################

trainer = autoregressive_train.AutoregressiveTrainer(
    model=model,
    data_loader=loader,
    params=trainer_params,
    snapshot_path=working_dir + '/snapshots',
    snapshot_name=args.run_name,
    snapshot_interval=args.num_iterations // 10,
    snapshot_exec_template=sbatch_executable,
    device=device,
    # logger=model_logging.Logger(validation_interval=None),
    logger=model_logging.TensorboardLogger(
        log_interval=500,
        validation_interval=None,
        generate_interval=5000,
        log_dir=working_dir + '/logs/' + args.run_name,
        print_output=True,
    )
)
if args.restore is not None:
    trainer.load_state(checkpoint)

print()
print("Model:", model.__class__.__name__)
print("Hyperparameters:", json.dumps(model.hyperparams, indent=4))
print("Trainer:", trainer.__class__.__name__)
print("Training parameters:", json.dumps(
    {key: value for key, value in trainer.params.items() if key != 'snapshot_exec_template'}, indent=4))
print("Dataset:", dataset.__class__.__name__)
print("Dataset parameters:", json.dumps(dataset.params, indent=4))
print("Num trainable parameters:", model.parameter_count())
print(f"Training for {args.num_iterations - model.step} iterations")

trainer.save_state()
trainer.train(steps=args.num_iterations)
