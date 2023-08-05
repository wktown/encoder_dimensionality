import os
import re
import argparse
import logging
import warnings
from typing import List, Dict, Any, Callable

import torch
import numpy as np
import pandas as pd
import xarray as xr
#from dnns.model_activations import yield_models, ModelWrapper, stimset_id
from neural_preprocessing import open_majajhong_assembly, preprocess_betas
from regression import CrossRegressionEvaluation
from read_file import read_stim_random

logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore", message=".*on tensors of dimension other than 2.*")

# default paths and parameters
#from utils.paths import *

from activation_models.generators import get_activation_models
from activation_models.AtlasNet.model_2L import EngineeredModel2L
def stimset_id(end_exclusive: int, stim_set_name: str = "imagenet_train") -> str:
    return (
        f"{stim_set_name}"
        if end_exclusive is None
        else f"{stim_set_name}_{end_exclusive}"
    )

model_list = {
    # 'model_final_test':EngineeredModel().Build(),
    # 'alexnet_test':torchvision.models.alexnet(pretrained=True),
    'alexnet_untrained_test_3':torchvision.models.alexnet(pretrained=False),
     }


class EncodingScore:
    def __init__(self, version: str):
        self._version = version

    def score(
        self,
        imageset: str,
        roi: str,
        regression: str,
        model_kwargs: Dict[str, Any] = {},
        #model_list: Callable = yield_models,
        model_list: Callable = get_activation_models(),
        start_idx: int = 0,
        num_yield: int = 1,
        **kwargs,
    ):
        path = f"/home/wtownle1/encoder_dimensionality/Ray_encoding/results/{self._version}/encoding_score"
        if not os.path.exists(path):
            os.mkdir(path)
        benchmark = f"{imageset}_{roi}_{regression}"
        if regression == "pls":
            ncomp = kwargs.get("n_components")
            benchmark = f"{benchmark}_ncomp={ncomp}"
        elif regression == "ridge":
            alpha = kwargs.get("regularization")
            benchmark = f"{benchmark}_alpha={alpha}"
        if not os.path.exists(f"{path}/{benchmark}"):
            os.mkdir(f"{path}/{benchmark}")
        self._benchmark_path = f"{path}/{benchmark}"
        for model in model_list(
            start_idx=start_idx, 
            num_yield=num_yield, 
            **model_kwargs
        ):
            if imageset == "nsd":
                scores = self._score_nsd(
                    model=model, 
                    roi=roi, 
                    regression=regression, 
                    **kwargs
                )
            elif imageset == "majajhong":
                scores = self._score_majajhong(
                    model=model, 
                    roi=roi, 
                    regression=regression,
                    **kwargs
                )

    # output aggregated value instead of raw
    def _score_majajhong(
        self,
        model: EngineeredModel2L,
        roi: str,
        regression: str,
        layers: List[str] = None,
        subject=None,
        **kwargs,
    ):
        if subject is None:
            subject = np.arange(2)
        elif isinstance(subject, int):
            subject = [subject]
        if layers is None:
            layers = model.layers
        full_stim = read_stim_random(
            data_dir= "/data/shared/brainio/brain-score/dicarlo.hvm-public",
            stim_format="png",
            num_stimuli=None,
            stim_set_name="hvm",
        )
        for layer in layers:
            id = f"{model.wrapper.identifier}_layer={layer}"
            logging.info(id)
            X = model.wrapper._extractor(
                stimuli=full_stim,
                layers=[layer],
                stimuli_identifier=stimset_id(end_exclusive=None, stim_set_name="hvm"),
            ).values
            full_stim_filename = [
                re.search(".*public/(.*)", s).group(1) for s in full_stim
            ]
            meta = pd.read_csv("/data/shared/brainio/brain-score/image_dicarlo_hvm-public.csv")
            id_name_dict = dict(zip(meta.image_id, meta.filename))
            object_name_dict = dict(zip(meta.image_id, meta.object_name))
            results, subjects = [], []
            for subj in subject:
                betas = open_majajhong_assembly(macaque=subj, roi=roi)
                subj_stim = betas.coords["image_id"].values
                idx = [full_stim_filename.index(id_name_dict[s]) for s in subj_stim]
                object_name = [object_name_dict[s] for s in subj_stim]
                subj_X, betas = X[idx], betas.values
                
                results.append(
                    CrossRegressionEvaluation(
                        regression=regression, **kwargs
                    )(subj_X, betas, object_name)['r']
                )
                subjects.extend([subj]*betas.shape[1])
            results = xr.DataArray(
                np.concatenate(results, axis=1),
                dims=["fold", "voxel"],
                coords={"subject": ("voxel", subjects)},
            )
            results.to_netcdf(f"{self._benchmark_path}/{model.wrapper.identifier}_layer={layer}")
            
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score neural encoding models")
    parser.add_argument("--imageset", type=str, default="nsd")
    parser.add_argument("--sidx", type=int, default=0)
    parser.add_argument("--nyield", type=int, default=1)
    parser.add_argument("--roi", type=str, default="general")
    parser.add_argument("--reg", type=str, default="pls")
    parser.add_argument("--alpha", type=int, default=1)
    parser.add_argument("--ncomp", type=int, default=25)
    parser.add_argument("--version", type=str)
    args = parser.parse_args()

    dv = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"ON device: {dv}")

    ES = EncodingScore(version=args.version)

    if args.reg == "pls":
        ES.score(
            imageset=args.imageset,
            roi=args.roi,
            regression=args.reg,
            start_idx=args.sidx,
            n_components=args.ncomp,
            num_yield=args.nyield,
        )
    elif args.reg == "ridge":
        ES.score(
            imageset=args.imageset,
            roi=args.roi,
            regression=args.reg,
            start_idx=args.sidx,
            regularization=args.alpha,
            num_yield=args.nyield,
        )
    else:
        ES.score(
            imageset=args.imageset,
            roi=args.roi,
            regression=args.reg,
            start_idx=args.sidx,
            num_yield=args.nyield,
        )
