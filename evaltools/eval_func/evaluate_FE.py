#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys

import numpy as np
import pandas as pd


def eval_FE_tl(df_gt_data, df_est):
    """
    Floor Error timeline

    Parameters
    ----------
    df_gt_data  : pandas.DataFrame [timestamp, x, y, theta, floor]
        ground truth
    df_est      : pandas.DataFrame [timestamp, x, y, floor, ...]
        estimated position

    Retruns
    ----------
    df_floor_error_tl : pandas.DataFrame [timestamp, type, ce]
    """
    df_gt_data = df_gt_data.dropna(subset=("x", "y"))
    df_est = df_est.dropna(subset=("x", "y"))
    df_eval = pd.merge_asof(df_gt_data, df_est,
                            left_index=True, right_index=True, tolerance=0.5, direction='nearest',
                            suffixes=["_gt", "_est"])
    df_eval = df_eval.dropna(subset=("x_gt", "y_gt", "x_est", "y_est"))

    # floor check
    df_eval["floor_correct"] = (df_eval["floor_est"] == df_eval["floor_gt"])

    df_fe_tl = pd.DataFrame(data={'timestamp': df_eval.index,
                            'type': 'fe', 'value': df_eval["floor_correct"]}).set_index('timestamp')

    return df_fe_tl


def eval_FE(df_est, step):
    """
    Calc CE
    (評価値まで算出する版。主に最適化用の入口)
    """
    if 'gt' not in step.keys():
        raise KeyError(F'[gt.data] is not found')
    gt = step['gt']['data']

    df_fe_tl = eval_FE_tl(gt, df_est)

    return df_fe_tl
