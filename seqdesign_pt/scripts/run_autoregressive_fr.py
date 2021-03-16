#!/usr/bin/env python3
import argparse
import os
import time
from datetime import timedelta
import sys
import json
import glob
import tensorflow as tf
import numpy as np
import torch
from seqdesign_pt.version import VERSION
from seqdesign_pt import data_loaders
from seqdesign_pt import autoregressive_train
from seqdesign_pt import autoregressive_model
from seqdesign_pt import model_logging
from seqdesign_pt import utils
from seqdesign_pt import aws_utils


def main(working_dir='.'):
    start_run_time = time.time()

    parser = argparse.ArgumentParser(description="Train an autoregressive model on a collection of sequences.")
    parser.add_argument("--s3-path", type=str, default='',
                        help="Base s3:// path (leave blank to disable syncing).")
    parser.add_argument("--s3-project", type=str, default=VERSION+'-pt', metavar='P',
                        help="Project name (subfolder of s3-path).")
    parser.add_argument("--run-name-prefix", type=str, default=None, metavar='P',
                        help="Prefix for run name.")
    parser.add_argument("--channels", type=int, default=48, metavar='C',
                        help="Number of channels.")
    parser.add_argument("--batch-size", type=int, default=30,
                        help="Batch size.")
    parser.add_argument("--num-iterations", type=int, default=250005, metavar='N',
                        help="Number of iterations to run the model.")
    parser.add_argument("--snapshot-interval", type=int, default=None, metavar='N',
                        help="Take a snapshot every N iterations.")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset name for fitting model. Alignment weights must be computed beforehand.")
    parser.add_argument("--restore", type=str, default='',
                        help="Session name for restoring a model to continue training.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="Which gpu to use. Usually  0, 1, 2, etc...")
    parser.add_argument("--r-seed", type=int, default=42, metavar='RSEED',
                        help="Random seed for parameter initialization")
    parser.add_argument("--dropout-p", type=float, default=0.5,
                        help="Dropout probability (drop rate, not keep rate)")
    parser.add_argument("--alphabet-type", type=str, default='protein', metavar='T',
                        help="Type of data to model. Options = [protein, DNA, RNA]")
    args = parser.parse_args()

    ########################
    # MAKE RUN DESCRIPTORS #
    ########################

    if args.restore == '':
        folder_time = (
            f"{args.dataset}_{args.s3_project}_channels-{args.channels}"
            f"_rseed-{args.r_seed}_{time.strftime('%y%b%d_%I%M%p', time.gmtime())}"
        )
        if args.run_name_prefix is not None:
            folder_time = args.run_name_prefix + '_' + folder_time
    else:
        folder_time = args.restore.split('/')[-1]
        folder_time = folder_time.split('.ckpt')[0]

    folder = f"{working_dir}/sess/{folder_time}"
    os.makedirs(folder, exist_ok=True)
    log_f = utils.Tee(f'{folder}/log.txt', 'a')  # log stdout to log.txt

    if args.dataset.endswith('.fa'):
        args.dataset = args.dataset[:-3]

    if not args.restore:
        restore_args = " \\\n  ".join(sys.argv[1:])
        restore_args += f" \\\n  --restore {{restore}}"
    else:
        restore_args = sys.argv[1:]
        restore_args[restore_args.index('--restore') + 1] = '{{restore}}'

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
      {restore_args}
    """

    ####################
    # SET RANDOM SEEDS #
    ####################

    if args.restore:
        # prevent from repeating batches/seed when restoring at intermediate point
        # script is repeatable as long as restored at same step
        # assumes restore arg of *.ckpt-(int step)
        restore_ckpt = args.restore.split('.ckpt-')[-1]
        r_seed = args.r_seed + int(restore_ckpt)
        r_seed = r_seed % (2 ** 32 - 1)  # limit of np.random.seed
    else:
        r_seed = args.r_seed

    tf.set_random_seed(r_seed)
    np.random.seed(args.r_seed)
    torch.manual_seed(args.r_seed)
    torch.cuda.manual_seed_all(args.r_seed)

    def _init_fn(worker_id):
        np.random.seed(args.r_seed + worker_id)

    #####################
    # PRINT SYSTEM INFO #
    #####################

    print(folder)
    print(args)

    print("OS: ", sys.platform)
    print("Python: ", sys.version)
    print("TensorFlow: ", tf.__version__)
    print("Numpy: ", np.__version__)

    use_cuda = not args.no_cuda
    device = torch.device(f"cuda:{args.gpu}" if use_cuda and torch.cuda.is_available() else "cpu")
    print('Using device:', device)
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(device))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(device)/1024**3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(device)/1024**3, 1), 'GB')
        print(utils.get_cuda_version())
        print("CuDNN Version ", utils.get_cudnn_version())

    print("SeqDesign git hash:", str(utils.get_github_head_hash()))
    print()

    print("Run:", folder_time)

    #############
    # LOAD DATA #
    #############

    aws_util = aws_utils.AWSUtility(s3_base_path=args.s3_path, s3_project=args.s3_project) if args.s3_path else None
    # for now, we will make all the sequences have the same length of
    #   encoded matrices, though this is wasteful
    filenames = glob.glob(f'{working_dir}/datasets/sequences/{args.dataset}*.fa')
    if not filenames and aws_util is not None:
        if not aws_util.s3_get_file_grep(
                s3_folder='datasets/sequences',
                dest_folder=f'{working_dir}/datasets/sequences/',
                search_pattern=f'{args.dataset}.*\\.fa',
        ):
            raise Exception("Could not download dataset files from S3.")
        filenames = glob.glob(f'{working_dir}/datasets/sequences/{args.dataset}*.fa')
    assert len(filenames) == 1

    dataset = data_loaders.SingleFamilyDataset(
        batch_size=args.batch_size,
        working_dir=working_dir,
        dataset=args.dataset,
        matching=True,
        unlimited_epoch=True,
        output_shape='NCHW',
        output_types='decoder',
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
        checkpoint = args.restore
        trainer_params = None
        dims = {"alphabet": len(dataset.alphabet)}
        model = autoregressive_model.AutoregressiveFR(channels=args.channels, dropout_p=args.dropout_p, dims=dims)
    model.to(device)

    ################
    # RUN TRAINING #
    ################

    trainer = autoregressive_train.AutoregressiveTrainer(
        model=model,
        data_loader=loader,
        params=trainer_params,
        snapshot_path=working_dir + '/sess',
        snapshot_name=folder_time,
        snapshot_interval=args.num_iterations // 10 if args.snapshot_interval is None else args.snapshot_interval,
        snapshot_exec_template=sbatch_executable,
        device=device,
        # logger=model_logging.Logger(validation_interval=None),
        logger=model_logging.TensorboardLogger(
            log_interval=500,
            validation_interval=1000,
            generate_interval=5000,
            log_dir=working_dir + '/logs/' + folder_time,
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

    print(f"Done! Total run time: {timedelta(seconds=time.time()-start_run_time)}")
    log_f.flush()
    if aws_util:
        aws_util.s3_sync(local_folder=folder, s3_folder=f'sess/{folder_time}/', destination='s3')

    if working_dir != '.':
        os.makedirs(f'{working_dir}/complete/', exist_ok=True)
        OUTPUT = open(f'{working_dir}/complete/{folder_time}.txt', 'w')
        OUTPUT.close()


if __name__ == "__main__":
    main()
