#!/bin/bash
# Example installation script of SeqDesign for Tensorflow-GPU from scratch
# Tested on Ubuntu 18.04 LTS, runtime ~5 minutes including a reboot.
# Miniconda and Tensorflow 1.12 are installed here, but a working Tensorflow 1 environment can substitute.
# Before running this script, first run `git clone -b v3 https://github.com/aaronkollasch/seqdesign-pytorch.git`
# and then `cd seqdesign-pytorch`
# If NVIDIA drivers have not been installed before, this script must be run twice, rebooting the system in between.

if [ ! -f "/proc/driver/nvidia/version" ]; then
  echo "NVIDIA driver not found; installing."
  sudo apt update
  sudo apt install -y --no-install-recommends nvidia-driver-430
  echo "
NVIDIA drivers installed.
Please reboot your system, then run linux_setup.sh a second time."
  exit
fi

# set up conda and the SeqDesign environment
if [ ! -d "$HOME/miniconda3" ]; then
  echo "miniconda3 not found; installing."
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  sh Miniconda3-latest-Linux-x86_64.sh -b -p "$HOME"/miniconda3
  rm Miniconda3-latest-Linux-x86_64.sh
fi
"$HOME"/miniconda3/bin/conda init
"$HOME"/miniconda3/bin/conda install mamba -n base -c conda-forge
"$HOME"/miniconda3/bin/mamba create -n seqdesign -y -c pytorch python=3.7 pip pytorch scipy scikit-learn gitpython pandas biopython pillow
"$HOME"/miniconda3/envs/seqdesign/bin/python -c "import torch; print(torch.cuda.is_available()); print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"  # test GPU install
"$HOME"/miniconda3/bin/conda install -y -n seqdesign "tensorflow>1.12,<2"  # necessary to read tensorflow model files
#"$HOME"/miniconda3/envs/seqdesign/bin/python -c "from tensorflow.python.client import device_lib; print(device_lib.list_local_devices())"  # test GPU install

# download SeqDesign code:
# git clone -b v3 https://github.com/aaronkollasch/seqdesign-pytorch.git
# cd seqdesign-pytorch || exit
"$HOME"/miniconda3/envs/seqdesign/bin/pip install .  # use setup.py develop if you want to modify the code files

# download demo/example data
if [ ! -f examples/datasets/sequences/BLAT_ECOLX_1_b0.5_lc_weights.fa ]; then
  "echo examples not found; downloading."
  cd examples || exit
  ./download_example_data.sh
fi

echo "
SeqDesign installed.
Run 'source ~/.bashrc; conda activate seqdesign' before using."

# # to run training demo:
# ./demo_train.sh

# # to run calc_logprobs using trained weights:
# ./demo_calc_logprobs.sh

# # to generate sequences:
# ./demo_generate.sh
