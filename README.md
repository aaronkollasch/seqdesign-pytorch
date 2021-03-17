# SeqDesign

SeqDesign is a generative, unsupervised model for biological sequences.
It is capable of learning functional constraints from unaligned sequences
in order to predict the effects of mutations and generate novel sequences,
including insertions and deletions. For more information,
check out the [biorxiv preprint](https://doi.org/10.1101/757252).

This version of the codebase is compatible with Python 3 and PyTorch.
It also implements [Fast Wavenet](https://github.com/tomlepaine/fast-wavenet) generation.  
A TensorFlow version is available [here](https://github.com/debbiemarkslab/SeqDesign)

## Installation

See [INSTALL.md](INSTALL.md).

## Examples

See the [examples](examples) directory.


## Usage
Run each script with the `-h` argument to see additional arguments:
### Training

Given a fasta file of training sequences, run:
```shell script
run_autoregressive_fr <your_dataset>.fa
```

### Mutation effect prediction
Deterministic:
```shell script
calc_logprobs_seqs_fr --sess <your_sess> --dropout-p 1.0 --num-samples 1 --input <input>.fa --output <output>.csv
```

Average of 500 samples:
```shell script
calc_logprobs_seqs_fr --sess <your_sess> --dropout-p 0.5 --num-samples 500 --input <input>.fa --output <output>.csv
```

### Sequence generation
```shell script
generate_sample_seqs_fr --sess <your_sess>
```
Use the `--fast-generation` argument for Fast Wavenet.

## Data availability
See the [examples](examples) directory to download training sequences,
mutation effect predictions, and generated sequences.
