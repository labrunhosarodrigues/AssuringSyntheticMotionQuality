# -*- coding: utf-8 -*-
"""
assets.__init__
---------------

motiongen assets to be packaged with code.

:copyright: (c) 2025 by Cognitive Systems Lab.
"""
# Imports

# built-in
from importlib import resources

# local

# 3rd-party


MASSIVE_PROMPTS = resources.path(
    __package__, "massive_prompts.txt"
)

GROUND_TRUTH_SCRIPT = resources.path(
    __package__, "generate_ground_truths.sh"
)

SINGLE_SAMPLE_VIDEO_SCRIPT = resources.path(
    __package__, "generate_single_samples.sh"
)

GENERATE_PILOT_SCRIPT = resources.path(
    __package__, "generate_pilot_data.sh"
)

VARIABILITY_GROUPS_SCRIPT = resources.path(
    __package__, "generate_variability_groups.sh"
)
