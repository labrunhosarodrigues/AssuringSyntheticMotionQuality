# -*- coding: utf-8 -*-
"""
motiongen.quality_checking
--------------------------

Script with quality checking of MDM generated motion samples.

:copyright: (c) 2024 by Cognitive Systems Lab.
:license: MIT
"""
# Imports
# built-in
import os
import math
import argparse

# local
import motiongen.data_handlers.data_handling as dh
import motiongen.motiongen.mdm_handling.visualization as visualization

# 3rd-party

# CSL


def get_sorted_scores(base_dir):
    """get pandas dataframe with samples sorted by ascending order of score.

    Parameters
    ----------
    base_dir : str
        path where to find the scores to sort.
    """

    scores = dh.aggregate_mobert_scores(base_dir)

    scores = scores.sort_values(by=["score"], ignore_index=True)
    scores["selected"] = True
    return scores


def clear_batch_folder(path):
    """Eliminate all contents from batch folder.

    Parameters
    ----------
    path : str
        Path to the batch folder.
    """

    _, _, files = next(os.walk(path))

    for file in files:
        os.remove(os.path.join(path, file))


def build_batch(scores, scores_range, data_path, save_path, batch_size=20):
    """Select the lowest scoring active samples to present
    an evaluation batch to a human evaluator.

    Parameters
    ----------
    scores : DataFrame
        dataframe containing the scores for each generated motion.
    scores_range : Slice
        Slice indicating what portion of the scores still needs to
        be evaluated.
    data_path : str
        directory where to find the motion data files.
    save_path : str
        directory where the selected batch is presented.
    batch_size : int, optional
        Size of batch to build, by default 20.
    """

    clear_batch_folder(save_path)

    center = (scores_range.stop - scores_range.start) // 2 + scores_range.start

    batch = scores.loc[scores.selected].iloc[
        center - (batch_size // 2):
        center + (batch_size // 2)
    ]

    for index, row in batch.iterrows():
        motion_type = row["motion"]
        motion_index = row["motion_index"]
        motion, text = dh.load_motion(
            os.path.join(data_path, motion_type),
            motion_index
        )
        animation_save_path = os.path.join(save_path, f"{index}_{text}.mp4")
        visualization.generate_animation(motion, text, animation_save_path)

    return batch


def register_removed(batch, scores, save_path):
    """Check what samples were deleted as non-accepted.

    Parameters
    ----------
    batch : DataFrame
        Dataframe listing batch samples
    save_path : str
        Path containing remaining videos of accepted samples from the batch.
    """

    _, _, files = next(os.walk(save_path))
    for file in files:
        name = os.path.splitext(file)[0]
        index, _ = name.split('_')
        batch.drop(index=int(index), inplace=True)

    for index, _ in batch.iterrows():
        scores.loc[index, 'selected'] = False

    removed_samples = len(batch)
    return removed_samples


def main(batch_size):
    scores = get_sorted_scores("./data/automatic_scores_large")
    scores_range = slice(0, len(scores))
    print(scores_range)

    while (scores_range.stop - scores_range.start) > batch_size:
        batch = build_batch(
            scores, scores_range,
            "./data/large_dataset",
            "current_batch",
            batch_size
        )
        input('press enter to continue after filtering batch...')
        removed_samples = register_removed(batch, scores, "current_batch")

        if (removed_samples / batch_size) < 0.2:
            # mostly good samples, move left
            scores_range = slice(scores_range.start, scores_range.stop // 2)
        elif (removed_samples / batch_size) > 0.6:
            # mostly bad samples, move right
            scores_range = slice(scores_range.stop // 2, scores_range.stop)
        else:  # found a balanced division, set separation there
            break

        print(scores_range)

    n_steps = (
        math.log2(len(scores)
                  / (scores_range.stop - scores_range.start))
        + 1
    )
    print(f"Number of needed iterations: {n_steps}")

    lower_bound = (
        (scores_range.stop - scores_range.start) // 2
        + scores_range.start
        - (batch_size // 2)
        - 1  # as .loc includes the last index
    )
    scores.loc[:lower_bound, "selected"] = False

    scores.to_csv("./data/large_dataset/filtering.csv")


def cli():
    parser = argparse.ArgumentParser(
        prog="quality_checker",
        description="Program to present batches of suspicious "
                    "samples for a user to filter out.",
        epilog="License MIT; Cognitive Systems Lab"
    )

    parser.add_argument(
        "-b",
        type=int, default=20,
        help="batch size to present to user. Default is 20."
    )

    args = parser.parse_args()

    main(args.b)


if __name__ == '__main__':
    cli()
