# -*- coding: utf-8 -*-
"""
pilot_generation.datasetcorrupter
---------------------------------

Module with functions and script to corrupt generated samples in order
to produce "poor quality" samples in terms of naturalness and faithfulness
to validate that MoBERT scores are capable of discriminating them from
properly generated samples.

:copyright: (c) 2024 by Cognitive Systems Lab.
:license: MIT
"""
# Imports

# built-in

# local
from mdmdataset import MDMGeneratedDataset

# 3rd-party
import numpy as np

# CSL


class MDMCorruptedDataset(MDMGeneratedDataset):

    def corrupt_data(self, processed_data):
        motion = processed_data[:120]
        new_motion = 12*np.random.permutation(motion)

        return new_motion

    def load_motion_data(self, path, params):
        feature_data = super().load_motion_data(path, params)
        for i, item in enumerate(feature_data):
            key, p_data, text = item
            feature_data[i] = (key, self.corrupt_data(p_data), text)

        return feature_data


if __name__ == "__main__":
    import yaml
    import pathlib
    from mobert.configs import BASE_CONFIG

    DATASETPATH = "motiongen/data/motions"

    with open(BASE_CONFIG, 'r') as fid:
        config = yaml.safe_load(fid)

    config = config["primary_evaluator"]
    primary_evaluator_model_config = config["primary_evaluator_model"]
    chunk_encoder_config = config["chunk_encoder"]
    tokenizer_and_embedders_config = config["tokenizer_and_embedders"]

    dataset = MDMCorruptedDataset(
        cache_path=pathlib.Path("."),
        path=pathlib.Path(DATASETPATH),
        chunk_size=chunk_encoder_config["chunk_size"],
        overlap=chunk_encoder_config["chunk_overlap"],
    )
    print(len(dataset))
