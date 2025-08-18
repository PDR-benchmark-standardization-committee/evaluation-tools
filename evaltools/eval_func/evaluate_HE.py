#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys

import numpy as np
import pandas as pd

from evaltools.com_tools.convert_tools import convert_quat_to_yaw


def eval_HE_tl(df_gt_data, df_est, right_on=False):
    """
    Calculate Heading-Error timeline

    Parameters
    ----------
    df_gt_data : pandas.DataFrame
        Ground truth, columns: [timestamp, x, y, theta, floor]
    df_est : pandas.DataFrame
        Estimated position, columns: [timestamp, x, y, floor, ...]
    right_on    : boolean
        False:pd.merge_asof(df_gt, df_est) / True:pd.merge_asof(df_est, df_gt)

    Returns
    -------
    result: pandas.DataFrame
        Error at each timestamp, columns: [timestamp, type, value]
    """
    type_tag = 'he'

    df_gt_data = df_gt_data.dropna(subset=("x", "y"))
    df_est = df_est.dropna(subset=("x", "y"))

    df_est = convert_quat_to_yaw(df_est)
    df_gt_data = convert_quat_to_yaw(df_gt_data)

    if right_on:
        df_eval = pd.merge_asof(df_est, df_gt_data,
                                left_index=True, right_index=True, tolerance=0.5, direction='nearest',
                                suffixes=["_est", "_gt"])
    else:
        df_eval = pd.merge_asof(df_gt_data, df_est,
                                left_index=True, right_index=True, tolerance=0.5, direction='nearest',
                                suffixes=["_gt", "_est"])
    df_eval = df_eval.dropna(subset=("yaw_est", "yaw_gt"))

    # # floor check
    # df_eval["floor_correct"] = (df_eval["floor_est"] == df_eval["floor_gt"])
    # df_eval_FC = df_eval[df_eval['floor_correct']]

    # calc error distance
    err_dst = df_eval['yaw_est'] - df_eval['yaw_gt']

    df_he_tl = pd.DataFrame(data={'timestamp': df_eval.index,
                            'type': type_tag, 'value': err_dst}).set_index('timestamp')

    return df_he_tl
