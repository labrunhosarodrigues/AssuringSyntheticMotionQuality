# -*- coding: utf-8 -*-
"""
pilot_generation.mobert_eval
----------------------------

Get MoBERT scores for Pilot data.

:copyright: (c) 2024 by Cognitive Systems Lab.
:license: MIT
"""
# Imports

# built-in
import os
import yaml
import pathlib
import argparse

# local
from motiongen.data_handlers.mdmdataset import MDMGeneratedDataset

# 3rd-party
import numpy as np
import torch
from mobert.models.motion_text_eval_bert import MotionTextEvalBERT
from mobert.configs import BASE_CONFIG
from mobert.primary_evaluator import (
    get_file_data,
    BEST_FAITHFULNESS_CHECKPOINT
)

# CSL


def load_config():
    with open(BASE_CONFIG, 'r') as fid:
        config = yaml.safe_load(fid)

    config = config["primary_evaluator"]
    primary_evaluator_model_config = config["primary_evaluator_model"]
    chunk_encoder_config = config["chunk_encoder"]
    tokenizer_and_embedders_config = config["tokenizer_and_embedders"]

    return (
        config, primary_evaluator_model_config,
        chunk_encoder_config,
        tokenizer_and_embedders_config
    )


def load_model(
    primary_evaluator_model_config,
    chunk_encoder_config,
    tokenizer_and_embedders_config,
    device
):
    # Create an instance of the model, with the path to tokenizer
    # and regressor checkpoints
    model = MotionTextEvalBERT(
        primary_evaluator_model_config,
        chunk_encoder_config,
        tokenizer_and_embedders_config,
        tokenizer_path=True,
        load_trained_regressors_path=True
    )
    # Send the model to the device
    model = model.to(device=device)
    # Load model checkpoint weights
    checkpoint = torch.load(
        get_file_data(BEST_FAITHFULNESS_CHECKPOINT),
        map_location=device
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    return model


def cli():
    parser = argparse.ArgumentParser(
        prog="mobert_eval.py",
        description="Run Mobert evaluation on path.",
        epilog="License: MIT; Cognitive Systems Lab."
    )

    parser.add_argument(
        "-i", "--input_dir",
        type=str,
        help="directory where to start looking for files to evaluate.",
        dest="in_dir"
    )

    parser.add_argument(
        "-o", "--out_dir",
        type=str,
        help="directory where to save evaluation results file.",
        dest="out_dir"
    )

    parser.add_argument(
        "-k", "--keep_cache",
        action="store_true",
        help="use this tag if you want to keep the cache of loaded datasets.",
        dest="keep_cache"
    )

    args = parser.parse_args()
    main(args.in_dir, args.out_dir, args.keep_cache)


def main(in_dir, out_dir, keep_cache):
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    (
        config, primary_evaluator_model_config,
        chunk_encoder_config,
        tokenizer_and_embedders_config
    ) = load_config()

    model = load_model(
        primary_evaluator_model_config,
        chunk_encoder_config,
        tokenizer_and_embedders_config,
        device
    )

    # Load motion and text data
    dataset = MDMGeneratedDataset(
        cache_path=pathlib.Path("."),
        path=pathlib.Path(in_dir),
        chunk_size=chunk_encoder_config["chunk_size"],
        overlap=chunk_encoder_config["chunk_overlap"],
    )

    faithfulness = []
    naturalness = []

    for item in dataset:

        # Motions should be a series of frame chunks (each 14 frames long)
        # with consecutive chunks having an overlap of 4.
        # Each frame should be represented in the 263-dimensional
        # representation as developed for HumanML3D.
        # Reference primary_evaluator_dataset.py for examples
        motions, motion_masks, texts = item.values()
        motions = motions.to(device=device).unsqueeze(0)
        motion_masks = motion_masks.to(device=device).unsqueeze(0)
        texts = [texts]
        # Alignment is the non-finetuned alignment probability prediction as
        # used in training and has gradients.
        # faithfulness_rating and naturalness_rating are human guidance
        # finetuned ratings using sklearn SVR regression models over the model
        # features (higher correlation than alignment).
        # All scores should range from [0, 1] with higher scores being better.
        # Regression scores may occur outside this range as well.
        (
            _, faithfulness_rating,
            naturalness_rating
        ) = model.rate_alignment_batch(
            texts, motions, motion_masks, device
        )
        faithfulness.append(faithfulness_rating)
        naturalness.append(naturalness_rating)

    data = (
        np.concatenate(faithfulness),
        np.concatenate(naturalness)
    )
    data = np.vstack(data).transpose((1, 0))

    with open(out_dir, "w") as fid:
        fid.write("faithfulness,naturalness,score\n")
        for a, b in data:
            fid.write(f"{a:.4f},{b:.4f},{a*b:.4f}\n")

    if not keep_cache:
        os.remove("./feat_data.obj")


if __name__ == "__main__":
    cli()
