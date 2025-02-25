# -*- coding: utf-8 -*-
"""
pilot_generation.data_handling
-------------

handling data loading and processing from MOBERT and WPD evaluation pilot.

:copyright: (c) 2024 by Cognitive Systems Lab.
:license: All rights reserved.
"""
# Imports

# built-in
import os
import json

# local

# 3rd-party
import numpy as np
import pandas as pd


def load_massive_prompts():
    prompts_path = "./data/large_dataset/massive_prompts.txt"
    with open(prompts_path, "r") as file:
        prompts = [line.rstrip().replace(' ', '_') for line in file]
    return prompts


def load_metadata(path):
    """Read JSON file with mapping between video file names
    and corresponding motion index in .npy files.
    """
    with open(path, "r") as fid:
        metadata = json.load(fid)

    return metadata


def load_mobert(path):
    """Load dataframe with naturalness and faithfulness scores
    obtained from MOBERT, ordered as the motions in the .npy files.
    """

    return pd.read_csv(path)


def load_result(path):
    """Load human annotations.
    """

    with open(path, "r") as fid:
        data = json.load(fid)

    return data


def load_motion(path, motion_index):
    """Load MDM motion from .npy file.

    Parameters
    ----------
    path : str
        path to .npy file.
    motion_index : int
        Index of motion to retrieve from file.
    """

    file_data = np.load(
        os.path.join(path, "results.npy"),
        allow_pickle=True)[()]
    motion = file_data.get('motion', [])[motion_index]
    motion = np.transpose(motion, (2, 0, 1))
    text = file_data.get('text', [])[motion_index]

    return motion, text


def reorganize_per_dimension(
        root, dimensions=["dimensions", "faithful", "variability"]):
    """
    Reorganize annotation results loaded from `root` directory
    into a structure that maps each annotator ID and measured dimension
    from `dimensions` to the corresponding DataFrame.
    Each produced dataframe will have columns:
     * `sample` - name of annotated video sample.
     * `{dimension_name}` - contains the annotation value.

    Parameters
    ----------
    root : str, path
        Directory where to load annotation results from.
        Must contain only the .JSON files intended to be loaded.
    dimensions : List[str], optional
        List of dimensions according to which the data will be
        organized. By default ["natural", "faithful", "variability"].

    Returns
    -------
    dict[str, dict[str, pd.DataFrame]]
        First level key is the annotator ID, second level key
        is the dimension.
    """
    _, _, files = next(os.walk(root))
    frames = {}

    for f in files:
        id = f.split('-')[0]
        data = load_result(os.path.join(root, f))
        df = pd.DataFrame([
            {'sample': k, **v} for k, v in data.items()
        ])

        frames[id] = {}
        for d in dimensions:
            frames[id][d] = df.loc[~df[d].isna(), ["sample", d]]

    return frames


def create_annotations_dataset(all_data, dimension):
    """
    Aggregate annotation data from a specific `dimension`.
    Results in a DataFrame with a column for each annotator's
    values for that dimension.
    Annotators' values are merged using an outer merge, missing
    values are inputed as `1.0` as annotators reported skiping
    annotation steps when they agreed with the default value
    of `1.0`.

    Parameters
    ----------
    all_data : dict
        With structure resulting from `reorganize_per_dimension`.
    dimension : str
        Dimension name to extract.
    Returns
    -------
    pd.DataFrame
    """

    dfs = pd.DataFrame([], columns=("sample", ))
    for k in all_data:
        df = all_data[k][dimension]
        df.columns = ("sample", k)
        dfs = dfs.merge(df, how="outer", on=["sample"])
    dfs = dfs.fillna(3.0)
    return dfs


def aggregate_mobert_scores(base_dir):
    _, _, files = next(os.walk(base_dir))

    data = []

    for file in files:
        df = load_mobert(os.path.join(base_dir, file))
        name = os.path.splitext(file)[0]
        df["motion"] = name
        df["motion_index"] = df.index
        data.append(df)

    return pd.concat(data, ignore_index=True)
