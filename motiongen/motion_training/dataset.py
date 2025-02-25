# -*- coding: utf-8 -*-
"""
motion_training.dataset
-----------------------

Module containing dataset definitions for training.

:copyright: Cognitive Systems Lab, 2025
"""
# Imports
# Built-in
import os

# Local

# 3r-party
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd


class MotionDataSet(Dataset):
    def __init__(self, data_path, references):
        """Initialize the Dataset object.

        Parameters
        ----------
        data_path : str
            Path to where the dataset files exist.
        references : pandas.DataFrame
            Dataframe containing the motion types and indices to reference
            the precise location of motion timeseries.
        """
        self.base = data_path
        self.references = references

    def __len__(self):
        return len(self.references)

    def __getitem__(self, idx):
        reference = self.references.iloc[idx]
        motion_type = reference["motion"]
        motion_index = reference["motion_index"]

        motion, _ = load_motion(
            os.path.join(self.base, motion_type),
            motion_index
        )

        motion = torch.Tensor(motion[:120])
        label = torch.zeros(10, dtype=torch.float32)
        label[reference["y"]] = 1.0

        return motion, label


def infect_data(clean, dirty):
    """Replace a random subset of clean data with the dirty samples."""
    n_replacement = len(dirty)
    drop_indices = np.random.choice(clean.index, n_replacement, replace=False)
    infected = pd.concat([clean.drop(drop_indices), dirty], ignore_index=True)

    return infected


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


def load_large_dataset(path):
    """Load Large dataset csv metadata"""
    dataset = pd.read_csv(path)
    prompts = load_massive_prompts()
    dataset["y"] = dataset["motion"].apply(
        lambda x: prompts.index(x) // 11
    )
    clean = dataset.loc[dataset["selected"]]
    dirty = dataset.loc[~dataset["selected"]]

    return clean, dirty


def load_massive_prompts():
    prompts_path = "./data/large_dataset/massive_prompts.txt"
    with open(prompts_path, "r") as file:
        prompts = [line.rstrip().replace(' ', '_') for line in file]
    return prompts
