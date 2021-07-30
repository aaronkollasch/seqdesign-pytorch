#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import time
import sys
import os
import argparse
import glob
import torch
import torch.distributions as dist
from seqdesign_pt import autoregressive_model
from seqdesign_pt import utils
from seqdesign_pt import aws_utils
from seqdesign_pt import data_loaders
from seqdesign_pt.tf_reader import TFReader
from seqdesign_pt.version import VERSION


def main():
    parser = argparse.ArgumentParser(description="Generate novel sequences sampled from the model.")
    parser.add_argument("--sess", type=str, required=True, help="Session name for restoring a model.")
    parser.add_argument("--checkpoint", type=int, default=None, metavar='CKPT', help="Checkpoint step number.")
    parser.add_argument("--channels", type=int, default=48, metavar='C', help="Number of channels.")
    parser.add_argument("--r-seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--temp", type=float, default=1.0, help="Generation temperature.")
    parser.add_argument("--batch-size", type=int, default=500, help="Number of sequences per generation batch.")
    parser.add_argument("--num-batches", type=int, default=1000000, help="Number of batches to generate.")
    parser.add_argument("--max-steps", type=int, default=50, help="Maximum number of decoding steps per batch.")
    parser.add_argument("--fast-generation", action='store_true', help="Use fast generation mode.")
    parser.add_argument("--input-seq", type=str, default='default', help="Path to file with starting sequence.")
    parser.add_argument("--output-prefix", type=str, default='nanobody', help="Prefix for output fasta file.")
    parser.add_argument("--alphabet-type", type=str, default='protein', metavar='T',  help="Alphabet to use for the dataset.", required=False)
    parser.add_argument("--s3-path", type=str, default='', help="Base s3:// path (leave blank to disable syncing).")
    parser.add_argument("--s3-project", type=str, default=VERSION, help="Project name (subfolder of s3-path).")
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

    print("SeqDesign-PyTorch git hash:", str(utils.get_github_head_hash()))
    print()

    aws_util = aws_utils.AWSUtility(s3_project=args.s3_project, s3_base_path=args.s3_path) if args.s3_path else None

    working_dir = "."

    # Variables for runtime modification
    sess_name = args.sess
    batch_size = args.batch_size
    num_batches = args.num_batches
    temp = args.temp
    r_seed = args.r_seed

    print(r_seed)

    np.random.seed(r_seed)
    torch.manual_seed(args.r_seed)
    torch.cuda.manual_seed_all(args.r_seed)

    dataset = data_loaders.SequenceDataset(alphabet_type=args.alphabet_type, output_types='decoder', matching=False)

    os.makedirs(os.path.join(working_dir, 'generate_sequences', 'generated'), exist_ok=True)
    output_filename = (
        f"{working_dir}/generate_sequences/generated/"
        f"{args.output_prefix}_start-{args.input_seq.split('/')[-1].split('.')[0]}"
        f"_temp-{temp}_param-{sess_name}_ckpt-{args.checkpoint}_rseed-{r_seed}.fa"
    )
    with open(output_filename, "w") as output:
        output.write('')

    # Provide the starting sequence to use for generation
    if args.input_seq != 'default':
        if not os.path.exists(args.input_seq) and aws_util:
            if '/' not in args.input_seq:
                args.input_seq = f'{working_dir}/generate_sequences/input/{args.input_seq}'
            aws_util.s3_get_file_grep(
                'generate_sequences/input',
                f'{working_dir}/generate_sequences/input',
                f"{args.input_seq.rsplit('/', 1)[-1]}"
            )
        with open(args.input_seq) as f:
            input_seq = f.read()
        input_seq = "*" + input_seq.strip()
    else:
        input_seq = "*EVQLVESGGGLVQAGGSLRLSCAASGFTFSSYAMGWYRQAPGKEREFVAAISWSGGSTYYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYC"

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
        model: autoregressive_model.AutoregressiveFR = autoregressive_model.AutoregressiveFR(
            channels=args.channels, dropout_p=0., hyperparams=hyperparams)
        reader.load_autoregressive_fr(model)
    else:
        checkpoint = torch.load(sess_namedir+'.pth', map_location='cpu')
        dims = checkpoint['model_dims']
        hyperparams = checkpoint['model_hyperparams']
        model: autoregressive_model.AutoregressiveFR = autoregressive_model.AutoregressiveFR(
            dims=dims, hyperparams=hyperparams, dropout_p=0.
        )
        model.load_state_dict(checkpoint['model_state_dict'])

    model: autoregressive_model.Autoregressive = model.model.model_f  # use only forward model
    model.to(device)
    print("Num parameters:", model.parameter_count())

    model.eval()
    for i in range(num_batches):
        start = time.time()

        input_seq_list = batch_size * [input_seq]
        batch = dataset.sequences_to_onehot(input_seq_list)
        seq_in = batch['decoder_input'].to(device)
        completion = torch.zeros(batch_size).to(device)

        for step in range(args.max_steps):
            with torch.no_grad():
                model.generate(args.fast_generation)
                seq_logits = model.forward(seq_in, None)
                output_logits = seq_logits[:, :, 0, -1] * args.temp
                output = dist.OneHotCategorical(logits=output_logits).sample()
                completion += output[:, dataset.aa_dict['*']]
                seq_in = torch.cat([seq_in, output.unsqueeze(-1).unsqueeze(-1)], dim=3)

            if (completion > 0).all():
                break

        batch_seqs = seq_in.argmax(1).squeeze().cpu().numpy()
        with open(output_filename, "a") as output:
            for idx_seq in range(batch_size):
                batch_seq = ''.join([dataset.idx_to_aa[idx] for idx in batch_seqs[idx_seq]])
                out_seq = ""
                end_seq = False
                for idx_aa, aa in enumerate(batch_seq):
                    if idx_aa != 0:
                        if end_seq is False:
                            out_seq += aa
                        if aa == "*":
                            end_seq = True
                output.write(f">{int(batch_size*i+idx_seq)}\n{out_seq}\n")

        print(f"Batch {i+1} done in {time.time()-start:0.4f} s")

    if aws_util:
        aws_util.s3_cp(
            local_file=output_filename,
            s3_file=f'generate_sequences/generated/{output_filename.rsplit("/", 1)[1]}',
            destination='s3'
        )


if __name__ == "__main__":
    main()
