# -*- coding: utf-8 -*-
"""
mdm_aux.__main__
----------------

Main script to use MDM generation through apython controled command interface.

:copyright: Cognitive Systems Lab, 2025
"""
# Imports
# Built-in
import argparse
import subprocess
import importlib

# Local
from mdm_aux import assets
from mdm_aux import generate_ground_truths

# 3r-party


def cli():
    prog = "mdm_aux"
    desc = "Auxiliary command line interface for MDM execution."
    epig = "(c) by Cognitive Systems Lab"

    parser = argparse.ArgumentParser(
        prog=prog,
        description=desc,
        epilog=epig
    )

    subparsers = parser.add_subparsers(dest="subcommand")

    subcommands = {
        "generate_pilot_data": gen_pilot_data,
        "generate_ground_truth_data": gen_ground_truth,
        "generate_variability_data": gen_variability,
        "generate_full_dataset": gen_full_dataset
    }

    subparsers.add_parser(
        "generate_pilot_data",
        help="Generate pilot samples from a prompt list."
    )

    subparsers.add_parser(
        "generate_ground_truth_data",
        help="Generate pilot samples from a HUMANML3D motion capture samples."
    )

    subparsers.add_parser(
        "generate_variability_data",
        help="Generate groupings of samples for the variability annotation."
    ).add_argument(
        "source",
        type=str,
        help="path to folder containing videos of samples to create groupings."
    )

    subparsers.add_parser(
        "generate_full_dataset",
        help="Generate Large dataset from massive prompts list."
    )

    args = parser.parse_args()

    subcommands[args.subcommand](args)


def gen_pilot_data(args):
    with importlib.resources.path(
        assets,
        "generate_pilot_data.sh"
    ) as script:
        subprocess.run(["bash", script], text=True, check=True)


def gen_ground_truth(args):
    generate_ground_truths.main()


def gen_variability(args):
    with importlib.resources.path(
        assets,
        "generate_variability_groups.sh"
    ) as script:
        subprocess.run(["bash", script, args.source], text=True, check=True)


def gen_full_dataset(args):
    with importlib.resources.path(
        assets,
        "generate_variability_groups.sh"
    ) as script:
        subprocess.run([
            "bash",
            script,
            "./data/large_dataset/massive_prompts.txt"
        ], text=True, check=True)


if __name__ == "__main__":
    cli()
