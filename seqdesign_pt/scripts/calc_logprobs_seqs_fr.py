#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import pandas as pd
import argparse
import sys
import os
import glob
import torch
from seqdesign_pt import autoregressive_model
from seqdesign_pt import autoregressive_train
from seqdesign_pt import utils
from seqdesign_pt import aws_utils
from seqdesign_pt import data_loaders
from seqdesign_pt.tf_reader import TFReader
from seqdesign_pt.version import VERSION


def main():
    working_dir = '..'

    parser = argparse.ArgumentParser(description="Calculate the log probability of mutated sequences.")
    parser.add_argument("--channels", type=int, default=48, metavar='C', help="Number of channels.")
    parser.add_argument("--r-seed", type=int, default=-1, metavar='RSEED', help="Random seed.")
    parser.add_argument("--num-samples", type=int, default=1, metavar='N', help="Number of iterations to run the model.")
    parser.add_argument("--minibatch-size", type=int, default=100, metavar='B', help="Minibatch size for inferring effect prediction.")
    parser.add_argument("--dropout-p", type=float, default=0., metavar='P', help="Dropout p while sampling log p(x).")
    parser.add_argument("--sess", type=str, default='', help="Session folder name for restoring a model.", required=True)
    parser.add_argument("--checkpoint", type=int, default=None,  metavar='CKPT', help="Checkpoint step number.")
    parser.add_argument("--input", type=str, default='',  help="Directory and filename of the input data.", required=True)
    parser.add_argument("--output", type=str, default='',  help="Directory and filename of the outout data.", required=True)
    parser.add_argument("--save-logits", action='store_true', help="Save logprobs matrices.")
    parser.add_argument("--save-ce", action='store_true',help="Save cross entropy matrices.")
    parser.add_argument("--alphabet-type", type=str, default='protein', metavar='T',  help="Alphabet to use for the dataset.", required=False)
    parser.add_argument("--s3-path", type=str, default='', help="Base s3:// path (leave blank to disable syncing).")
    parser.add_argument("--s3-project", type=str, default=VERSION+'-pt', help="Project name (subfolder of s3-path).")
    parser.add_argument("--num-data-workers", type=int, default=0, help="Number of workers to load data")
    parser.add_argument("--no-cuda", action='store_true', help="Disable GPU evaluation")
    parser.add_argument("--from-tf", action='store_true', help="Load model from tensorflow checkpoint")

    args = parser.parse_args()

    print(args)

    print('Call:', ' '.join(sys.argv))
    print("OS: ", sys.platform)
    print("Python: ", sys.version)
    print("PyTorch: ", torch.__version__)
    print("TensorFlow: ", tf.__version__)
    print("Numpy: ", np.__version__)

    use_cuda = not args.no_cuda
    device = torch.device("cuda:0" if use_cuda and torch.cuda.is_available() else "cpu")
    print('Using device:', device)
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3, 1), 'GB')
        print(utils.get_cuda_version())
        print("CuDNN Version ", utils.get_cudnn_version())

    print("SeqDesign git hash:", str(utils.get_github_head_hash()))
    print()

    sess_name = args.sess
    input_filename = args.input
    output_filename = args.output

    aws_util = aws_utils.AWSUtility(s3_project=args.s3_project, s3_base_path=args.s3_path) if args.s3_path else None
    if not os.path.exists(input_filename) and input_filename.startswith('input/') and aws_util is not None:
        folder, filename = input_filename.rsplit('/', 1)
        if not aws_util.s3_get_file_grep(
                s3_folder=f'calc_logprobs/{folder}',
                dest_folder=f'{working_dir}/calc_logprobs/{folder}',
                search_pattern=f'{filename}',
        ):
            raise Exception("Could not download test data from S3.")

    dataset = data_loaders.FastaDataset(
        batch_size=args.minibatch_size,
        working_dir='.',
        dataset=args.input,
        matching=True,
        unlimited_epoch=False,
        alphabet_type=args.alphabet_type,
    )
    loader = data_loaders.GeneratorDataLoader(
        dataset,
        num_workers=args.num_data_workers,
        pin_memory=True,
    )
    print("Read in test data.")

    if args.checkpoint is None:  # look for old-style flat session file structure
        glob_path = f"{working_dir}/sess/{sess_name}*"
        grep_path = f'{sess_name}.*'
        sess_namedir = f"{working_dir}/sess/{sess_name}"
    else:  # look for new folder-based session file structure
        glob_path = f"{working_dir}/sess/{sess_name}/{sess_name}.ckpt-{args.checkpoint}*"
        grep_path = f'{sess_name}.ckpt-{args.checkpoint}.*'
        sess_namedir = f"{working_dir}/sess/{sess_name}/{sess_name}.ckpt-{args.checkpoint}"

    if not glob.glob(glob_path) and aws_util:
        if not aws_util.s3_get_file_grep(
                f'sess/{sess_name}',
                f'{working_dir}/sess/{sess_name}',
                grep_path,
        ):
            raise Exception("Could not download session files from S3.")

    print("Initializing and loading variables")
    if args.from_tf:
        reader = TFReader(sess_namedir)
        legacy_version = reader.get_checkpoint_legacy_version()
        last_dilation_size = 200 if legacy_version == 0 else 256
        hyperparams = {'encoder': {'dilation_schedule': [1, 2, 4, 8, 16, 32, 64, 128, last_dilation_size],
                                   "config": "original",
                                   'dropout_type': 'independent'}}
        model = autoregressive_model.AutoregressiveFR(
            channels=args.channels, dropout_p=1-args.dropout_p, hyperparams=hyperparams)
        reader.load_autoregressive_fr(model)
    else:
        checkpoint = torch.load(sess_namedir+'.pth', map_location='cpu')
        dims = checkpoint['model_dims']
        hyperparams = checkpoint['model_hyperparams']
        model = autoregressive_model.AutoregressiveFR(dims=dims, hyperparams=hyperparams, dropout_p=1-args.dropout_p)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    print("Num parameters:", model.parameter_count())

    trainer = autoregressive_train.AutoregressiveTrainer(
        model=model,
        data_loader=None,
        device=device,
    )
    output = trainer.test(loader, model_eval=False, num_samples=args.num_samples, return_logits=args.save_logits, return_ce=args.save_ce)
    if args.save_logits or args.save_ce:
        output, logits = output
        logits_path = os.path.splitext(args.output)[0]
        os.makedirs(logits_path, exist_ok=True)
        for key, value in logits.items():
            np.save(f"{logits_path}/{key}.npy", value)

    os.makedirs(output_filename.rsplit('/', 1)[0], exist_ok=True)
    output = pd.DataFrame(output, columns=output.keys())
    output.to_csv(args.output, index=False)
    print("Done!")

    if output_filename.startswith('output/') and aws_util:
        aws_util.s3_cp(
            local_file=output_filename,
            s3_file=f'calc_logprobs/output/{output_filename.replace("output/", "")}',
            destination='s3'
        )


if __name__ == "__main__":
    main()
