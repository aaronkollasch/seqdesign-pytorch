import os
import sys
import subprocess
import glob
import collections
import shutil
import contextlib

import numpy as np
import torch


def recursive_update(orig_dict, update_dict):
    """Update the contents of orig_dict with update_dict"""
    for key, val in update_dict.items():
        if isinstance(val, collections.Mapping):
            orig_dict[key] = recursive_update(orig_dict.get(key, {}), val)
        else:
            orig_dict[key] = val
    return orig_dict


def comb_losses(losses_f, losses_r):
    losses_comb = {}
    for key in losses_f.keys():
        if 'per_seq' in key:
            losses_comb[key] = torch.stack([losses_f[key], losses_r[key]])
        else:
            losses_comb[key] = losses_f[key] + losses_r[key]
            losses_comb[key + '_f'] = losses_f[key]
            losses_comb[key + '_r'] = losses_r[key]
    return losses_comb


# https://stackoverflow.com/questions/49555991/can-i-create-a-local-numpy-random-seed
@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


# https://github.com/ilkarman/DeepLearningFrameworks/blob/master/notebooks/common/utils.py
def get_gpu_name():
    try:
        out_str = subprocess.run(["nvidia-smi", "--query-gpu=gpu_name", "--format=csv"], stdout=subprocess.PIPE).stdout
        out_list = out_str.decode("utf-8").split('\n')
        out_list = out_list[1:-1]
        return out_list
    except Exception as e:
        print(e)


def get_cuda_path():
    nvcc_path = shutil.which('nvcc')
    if nvcc_path is not None:
        return nvcc_path.replace('bin/nvcc', '')
    else:
        return None


# https://github.com/ilkarman/DeepLearningFrameworks/blob/master/notebooks/common/utils.py
def get_cuda_version():
    """Get CUDA version"""
    path = get_cuda_path() + 'version.txt'
    if path is not None and os.path.isfile(path):
        with open(path, 'r') as f:
            data = f.read().replace('\n', '')
        return data
    else:
        return "No CUDA in this machine"


# https://github.com/ilkarman/DeepLearningFrameworks/blob/master/notebooks/common/utils.py
def get_cudnn_version():
    """Get CUDNN version"""
    if sys.platform == 'win32':
        raise NotImplementedError("Implement this!")
        # This breaks on linux:
        # cuda=!ls "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
        # candidates = ["C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\" + str(cuda[0]) +"\\include\\cudnn.h"]
    elif sys.platform == 'linux':
        candidates = ['/usr/include/x86_64-linux-gnu/cudnn_v[0-99].h',
                      '/usr/local/cuda/include/cudnn.h',
                      '/usr/include/cudnn.h']
    elif sys.platform == 'darwin':
        candidates = ['/usr/local/cuda/include/cudnn.h',
                      '/usr/include/cudnn.h']
    else:
        raise ValueError("Not in Windows, Linux or Mac")
    cuda_path = get_cuda_path()
    if cuda_path is not None:
        candidates.append(cuda_path + 'include/cudnn.h')
    file = None
    for c in candidates:
        file = glob.glob(c)
        if file:
            break
    if file:
        with open(file[0], 'r') as f:
            version = ''
            for line in f:
                if "#define CUDNN_MAJOR" in line:
                    version = line.split()[-1]
                if "#define CUDNN_MINOR" in line:
                    version += '.' + line.split()[-1]
                if "#define CUDNN_PATCHLEVEL" in line:
                    version += '.' + line.split()[-1]
        if version:
            return version
        else:
            return "Cannot find CUDNN version"
    else:
        return "No CUDNN in this machine"
