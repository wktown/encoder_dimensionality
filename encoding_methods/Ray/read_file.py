import os
import re
from typing import List

import numpy as np


# default paths and parameters
from utils.paths import *


def read_file_general(data_dir: str, file_format: str) -> List[str]:
    file_list = []
    for root, _, files in os.walk(data_dir):
        file_list.extend(
            [os.path.join(root, x) for x in files if (x.endswith("." + file_format))]
        )
    return file_list


def read_directory(data_dir: str) -> List[str]:
    return [os.path.join(data_dir, name) for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))]

# hard-coded for the three relevant imagesets
def read_stim_random(
    data_dir: str,
    stim_format: str,
    num_stimuli: str,
    stim_set_name: str,
    random_seed: int = 11,
) -> List[str]:
    ref_file = f"{REF_DIR}/{stim_set_name}_num_stimuli={num_stimuli}_seed={random_seed}.npy"
    if os.path.exists(ref_file):
        file_list = np.load(ref_file)
        file_list = [f"{REF_DICT[stim_set_name]}/{fl}" for fl in file_list]
        return file_list
    rng = np.random.default_rng(random_seed)
    if stim_set_name == "imagenet_train":
        file_list = []
        stim_dict = {}
        subdir_list = [
            os.path.join(data_dir, x)
            for x in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, x))
        ]
        for sd in subdir_list:
            stim_dict[sd] = np.array(
                [
                    os.path.join(sd, x)
                    for x in os.listdir(sd)
                    if x.endswith("." + stim_format)
                ]
            )
        num_cat = len(stim_dict.keys())
        num_per_cat = [num_stimuli // num_cat] * num_cat
        for i in range(num_stimuli % num_cat):
            num_per_cat[i] += 1
        rng.shuffle(num_per_cat)
        for i, key in enumerate(stim_dict.keys()):
            idx = rng.choice(len(stim_dict[key]), size=num_per_cat[i], replace=False)
            file_list.extend(stim_dict[key][idx])
        rng.shuffle(file_list)
        fl = np.array([re.search(".*shared/(.*)", f).group(1) for f in file_list])
        np.save(ref_file, fl)
    else:  # nsd or hvm
        file_list = np.array(
            [
                os.path.join(data_dir, x)
                for x in os.listdir(data_dir)
                if x.endswith("." + stim_format)
            ]
        )
        rng.shuffle(file_list)
        if num_stimuli is not None:
            file_list = file_list[:num_stimuli]
        fl = np.array([re.search(".*shared/(.*)", f).group(1) for f in file_list])
        np.save(ref_file, fl)
    return file_list
