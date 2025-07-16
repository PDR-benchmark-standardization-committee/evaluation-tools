#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys

import numpy as np
import pandas as pd


def eval_CE_tl(df_gt_data, df_est, right_on=False):
    """
    Calculate Circular-Error timeline

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
    type_tag = 'ce'

    df_gt_data = df_gt_data.dropna(subset=("x", "y"))
    df_est = df_est.dropna(subset=("x", "y"))

    if right_on:
        df_eval = pd.merge_asof(df_est, df_gt_data,
                                left_index=True, right_index=True, tolerance=0.5, direction='nearest',
                                suffixes=["_est", "_gt"])
    else:
        df_eval = pd.merge_asof(df_gt_data, df_est,
                                left_index=True, right_index=True, tolerance=0.5, direction='nearest',
                                suffixes=["_gt", "_est"])
    df_eval = df_eval.dropna(subset=("x_gt", "y_gt", "x_est", "y_est"))

    # # floor check
    # df_eval["floor_correct"] = (df_eval["floor_est"] == df_eval["floor_gt"])
    # df_eval_FC = df_eval[df_eval['floor_correct']]

    # calc error distance
    err_dst = np.sqrt((df_eval['x_gt'] - df_eval['x_est'])
                      ** 2 + (df_eval['y_gt'] - df_eval['y_est'])**2)

    df_ce_tl = pd.DataFrame(data={'timestamp': df_eval.index,
                            'type': type_tag, 'value': err_dst}).set_index('timestamp')

    return df_ce_tl


def eval_SE_tl(df_gt_data, df_est):
    """
    Calculate Sphere-Error timeline

    Parameters
    ----------
    df_gt_data  : pandas.DataFrame [timestamp, x, y, z, (floor)]
        ground truth
    df_est      : pandas.DataFrame [timestamp, x, y, z, (floor), ...]
        estimated position

    Retruns
    ----------
    df_error_distance_tl : pandas.DataFrame [timestamp, type, ce]
        timeline ce = distance error
    """
    type_tag = 'se'

    df_gt_data = df_gt_data.dropna(subset=("x", "y", "z"))
    df_est = df_est.dropna(subset=("x", "y", "z"))
    df_eval = pd.merge_asof(df_gt_data, df_est,
                            left_index=True, right_index=True, tolerance=0.5, direction='nearest',
                            suffixes=["_gt", "_est"])
    df_eval = df_eval.dropna(
        subset=("x_gt", "y_gt", "z_gt", "x_est", "y_est", "z_est"))

    # floor check
    if 'floor' in df_gt_data.columns and 'floor' in df_est.columns:
        df_eval["floor_correct"] = (
            df_eval["floor_est"] == df_eval["floor_gt"])
        df_eval_FC = df_eval[df_eval['floor_correct']]

    # calc error distance
    err_dst = np.sqrt((df_eval['x_gt'] - df_eval['x_est'])
                      ** 2 + (df_eval['y_gt'] - df_eval['y_est'])**2 + (df_eval['z_gt'] - df_eval['z_est'])**2)

    df_ce_tl = pd.DataFrame(data={'timestamp': df_eval.index,
                            'type': type_tag, 'value': err_dst}).set_index('timestamp')

    return df_ce_tl


def eval_CE(df_est, step, quantile=50):
    """
    Entry point for integrated localization in step format
    """
    if 'gt' not in step.keys():
        raise KeyError(F'[gt.data] is not found')
    gt = step['gt']['data']

    df_ce_tl = eval_CE_tl(gt, df_est)

    return np.percentile(df_ce_tl.value, quantile)
