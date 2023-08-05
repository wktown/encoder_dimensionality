import os
import argparse
import logging
from typing import Set, Any
from collections.abc import Hashable
from pathlib import Path

import torch
import numpy as np
import xarray as xr
from brainio.assemblies import DataAssembly
from brainscore.benchmarks._neural_common import average_repetition

logging.basicConfig(level=logging.INFO)


# default paths and parameters
from utils.paths import *

# the rest is mostly adapted from raj's code
IDENTIFIER = "allen2021.natural_scenes"
BUCKET_NAME = "natural-scenes-dataset"
BIBTEX = """
@article{Allen2021,
    doi = {10.1038/s41593-021-00962-x},
    url = {https://doi.org/10.1038/s41593-021-00962-x},
    year = {2021},
    month = dec,
    publisher = {Springer Science and Business Media {LLC}},
    volume = {25},
    number = {1},
    pages = {116--126},
    author = {Emily J. Allen and Ghislain St-Yves and Yihan Wu and Jesse L. Breedlove and Jacob S. Prince and Logan T. Dowdle and Matthias Nau and Brad Caron and Franco Pestilli and Ian Charest and J. Benjamin Hutchinson and Thomas Naselaris and Kendrick Kay},
    title = {A massive 7T {fMRI} dataset to bridge cognitive neuroscience and artificial intelligence},
    journal = {Nature Neuroscience}
}
"""
ROIS = {
    "general": ({"source": "nsdgeneral", "label": "nsdgeneral"},),
    "faces": ({"source": "floc-faces"},),
    "words": ({"source": "floc-words"},),
    "places": ({"source": "floc-places"},),
    "bodies": ({"source": "floc-bodies"},),
    "V1-4": ({"source": "prf-visualrois"},),
    "V4": ({"source": "prf-visualrois", "label": "hV4"},),
    "V3": (
        {"source": "prf-visualrois", "label": "V3v"},
        {"source": "prf-visualrois", "label": "V3d"},
    ),
    "V2": (
        {"source": "prf-visualrois", "label": "V2v"},
        {"source": "prf-visualrois", "label": "V2d"},
    ),
    "V1": (
        {"source": "prf-visualrois", "label": "V1v"},
        {"source": "prf-visualrois", "label": "V1d"},
    ),
    "ventral": (
        {"source": "streams", "label": "ventral"},
        # {"source": "streams", "label": "midventral"}, 
    ),
    "lateral": (
        {"source": "streams", "label": "lateral"},
        # {"source": "streams", "label": "midlateral"},
    ),
    "parietal": (
        {"source": "streams", "label": "parietal"},
        # {"source": "streams", "label": "midparietal"},
    ),
    "frontal": (
        {"source": "corticalsulc", "label": "SFRS"},
        {"source": "corticalsulc", "label": "MFG"},
        {"source": "corticalsulc", "label": "IFG"},
    )
}
MACAQUES = {0: "Chabo", 1: "Tito"}


def open_majajhong_assembly(macaque: int, roi: str) -> xr.DataArray:
    assembly = DataAssembly(xr.open_dataarray(MH_PATH))
    assembly = average_repetition(assembly)
    assembly = (
        assembly.rename("data")
        .assign_attrs(
            {
                "time_bin": (
                    assembly["time_bin_start"].values[0],
                    assembly["time_bin_end"].values[0],
                ),
                "animal": MACAQUES[macaque],
            }
        )
        .isel(time_bin=0, drop=True)
        .isel({"neuroid": assembly["animal"].values == MACAQUES[macaque]})
    )
    assembly = assembly.isel({"neuroid": assembly["region"].values == roi}).transpose(
        "presentation", "neuroid"
    )
    return assembly


## mostly adapted from Raj Magesh's code
def open_subject_assembly(
    subject: int, *, filepath: Path = NSD_PATH, **kwargs
) -> xr.Dataset:
    return xr.open_dataset(filepath, group=f"/subject-{subject}", **kwargs)


def compute_shared_stimulus_ids(assemblies) -> Set[str]:
    return set.intersection(
        *(set(assembly["stimulus_id"].values) for assembly in assemblies)
    )


def compute_noise_ceiling(assembly: xr.Dataset) -> xr.DataArray:
    groupby = assembly["stimulus_id"].groupby("stimulus_id")
    counts = np.array([len(reps) for reps in groupby.groups.values()])
    if counts is None:
        fraction = 1
    else:
        unique, counts = np.unique(counts, return_counts=True)
        reps = dict(zip(unique, counts))
        fraction = (reps[1] + reps[2] / 2 + reps[3] / 3) / (reps[1] + reps[2] + reps[3])
    ncsnr_squared = assembly["ncsnr"] ** 2
    return (ncsnr_squared / (ncsnr_squared + fraction)).rename("noise ceiling")


def z_score_betas_within_sessions(betas: xr.DataArray) -> xr.DataArray:
    def z_score(betas: xr.DataArray) -> xr.DataArray:
        mean = betas.mean("presentation")
        std = betas.std("presentation")
        return (betas - mean) / std

    return (
        betas.load()
        .groupby("session_id")
        .map(func=z_score, shortcut=True)
        .assign_attrs(betas.attrs)
        .rename(betas.name)
    )


def z_score_betas_within_runs(betas: xr.DataArray) -> xr.DataArray:
    # even-numbered trials (i.e. Python indices 1, 3, 5, ...) had 62 trials
    # odd-numbered trials (i.e. Python indices 0, 2, 4, ...) had 63 trials
    n_runs_per_session = 12
    n_sessions = len(np.unique(betas["session_id"]))
    run_id = []
    for i_session in range(n_sessions):
        for i_run in range(n_runs_per_session):
            n_trials = 63 if i_run % 2 == 0 else 62
            run_id.extend([i_run + i_session * n_runs_per_session] * n_trials)
    betas["run_id"] = ("presentation", run_id)

    def z_score(betas: xr.DataArray) -> xr.DataArray:
        mean = betas.mean("presentation")
        std = betas.std("presentation")
        return (betas - mean) / std

    return (
        betas.load()
        .groupby("run_id")
        .map(func=z_score, shortcut=True)
        .assign_attrs(betas.attrs)
        .rename(betas.name)
    )


def remove_invalid_voxels_from_betas(
    betas: xr.DataArray, validity: xr.DataArray
) -> xr.DataArray:
    neuroid_filter = np.all(
        validity.stack(dimensions={"neuroid": ("x_", "y_", "z_")})[:-1, :],
        axis=0,
    )
    neuroid_filter = neuroid_filter.isel({"neuroid": neuroid_filter})
    valid_voxels = set(neuroid_filter.indexes["neuroid"].values)
    index = betas.set_index({"neuroid": ("x", "y", "z")}).indexes["neuroid"].values
    betas = betas.isel({"neuroid": [voxel in valid_voxels for voxel in index]})
    return betas


def groupby_reset(
    x: xr.DataArray, *, groupby_coord: str, groupby_dim: Hashable
) -> xr.DataArray:
    return (
        x.reset_index(groupby_coord)
        .rename({groupby_coord: groupby_dim})
        .assign_coords({groupby_coord: (groupby_dim, x[groupby_coord].values)})
        .drop_vars(f"{groupby_coord}_")
    )


def average_betas_across_reps(betas: xr.DataArray) -> xr.DataArray:
    return groupby_reset(
        betas.load()
        .groupby("stimulus_id")
        .mean()
        .assign_attrs(betas.attrs)
        .rename(betas.name),
        groupby_coord="stimulus_id",
        groupby_dim="presentation",
    ).transpose("neuroid", "presentation")


def filter_betas_by_roi(
    betas: xr.DataArray,
    *,
    rois: xr.DataArray,
    selectors,
) -> xr.DataArray:
    rois = rois.load().set_index({"roi": ("source", "label", "hemisphere")})
    selections = []
    for selector in selectors:
        selection = rois.sel(selector).values
        if selection.ndim == 1:
            selection = np.expand_dims(selection, axis=0)
        selections.append(selection)
    selection = np.any(np.concatenate(selections, axis=0), axis=0)
    betas = betas.load().isel({"neuroid": selection})
    return betas


def filter_betas_by_stimulus_id(
    betas: xr.DataArray, *, stimulus_ids: Set[str], exclude: bool = False
) -> xr.DataArray:
    selection = np.isin(betas["stimulus_id"].values, list(stimulus_ids))
    if exclude:
        selection = ~selection
    return betas.isel({"presentation": selection})


def open_betas_by_roi(
    subject: int,
    roi: str,
    filepath: str = NSD_PATH,
) -> xr.DataArray:
    filename = f"identifier={IDENTIFIER}_roi={roi}_subject={subject}"
    if not os.path.exists(f"{BETA_CACHE}/{filename}.nc"):
        assembly = open_subject_assembly(subject=subject, filepath=filepath)
        betas = filter_betas_by_roi(
            betas=assembly["betas"],
            rois=assembly["rois"],
            selectors=ROIS[roi],
        )
    else:
        betas = xr.open_dataarray(f"{BETA_CACHE}/{filename}.nc")
    return remove_invalid_voxels_from_betas(betas, validity=assembly["validity"])


def preprocess_betas(
    *,
    subject: int,
    roi: str,
    filepath: str = NSD_PATH,
    z_score: str = "session",
    average_across_reps: bool = True,
    **kwargs: Any,
) -> xr.DataArray:
    filename = f"identifier={IDENTIFIER}_roi={roi}_z_score={z_score}_average_across_reps={average_across_reps}_subject={subject}"
    if not os.path.exists(f"{BETA_CACHE}/{filename}.nc"):
        betas = open_betas_by_roi(
            filepath=filepath,
            subject=subject,
            roi=roi,
        ).load()
        if z_score is not None:
            if z_score == "session":
                betas = z_score_betas_within_sessions(betas)
            elif z_score == "run":
                betas = z_score_betas_within_runs(betas)
            else:
                raise ValueError("z_score must be 'session', 'run', or None")

        if average_across_reps:
            betas = average_betas_across_reps(betas)
        else:
            reps: dict[str, int] = {}
            rep_id: list[int] = []
            for stimulus_id in betas["stimulus_id"].values:
                if stimulus_id in reps:
                    reps[stimulus_id] += 1
                else:
                    reps[stimulus_id] = 0
                rep_id.append(reps[stimulus_id])
            betas = betas.assign_coords({"rep_id": ("presentation", rep_id)})

        betas = betas.rename(
            f"identifier={IDENTIFIER}_roi={roi}.z_score={z_score}_average_across_reps={average_across_reps}_subject={subject}"
        ).transpose("presentation", "neuroid")
        # may not needed for nsd general as well
        # if roi == "general":
        betas.to_netcdf(f"{BETA_CACHE}/{filename}.nc")
    else:
        betas = xr.open_dataarray(f"{BETA_CACHE}/{filename}.nc")
    return betas


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get betas")
    parser.add_argument("--subject", type=int)
    parser.add_argument("--roi", type=str)
    args = parser.parse_args()
    
    if not os.path.exists(BETA_CACHE):
        os.mkdir(BETA_CACHE)
    
    _ = preprocess_betas(
        subject=args.subject,
        roi=args.roi
    )

