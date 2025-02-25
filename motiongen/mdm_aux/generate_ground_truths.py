# -*- coding: utf-8 -*-
"""
mdm_aux.generate_ground_truths
------------------------------

Generate animations from actual motion capture recordings in HumanML3D.

:copyright: Cognitive Systems Lab, 2025
"""
# Imports
# Built-in
import sys
sys.path.append("../mdm")

# Local
from data_loaders.humanml.utils.plot_script import plot_3d_motion
from data_loaders.humanml.utils.paramUtil import t2m_kinematic_chain as skeleton

# 3r-party
import numpy as np


def main():
    files = ['000016', '000003', '010942', 'M014129', '005360']

    for f in files:
        joints = np.load(f'../HumanML3D/HumanML3D/new_joints/{f}.npy')
        title = ""
        dataset = "HumanML3D"
        fps = 20
        save_path = f"../sampleGT_rep{f[-2:]}.mp4"

        plot_3d_motion(save_path, skeleton, joints, title, dataset, fps=fps)