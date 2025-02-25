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

# local
import motiongen.motiongen.mdm_handling.plot_script as ps
from motiongen.motiongen.mdm_handling.paramUtil import (
    kit_kinematic_chain,
    t2m_kinematic_chain
)

# 3rd-party
import plotly.graph_objects as go

# CSL


def generate_animation(motion, caption, save_path):
    dataset = 'HumanML3D'
    fps = 12.5 if dataset == 'kit' else 20
    skeleton = kit_kinematic_chain if dataset == 'kit' else t2m_kinematic_chain

    ps.plot_3d_motion(
        save_path,
        skeleton, motion, dataset=dataset,
        title=caption, fps=fps
    )


def visualize_scores(scores):
    fig = go.Figure(go.Scatter(
        x=scores.naturalness,
        y=scores.faithfulness,
        marker_color=scores.score,
        mode="markers"
    ))

    fig.show()
