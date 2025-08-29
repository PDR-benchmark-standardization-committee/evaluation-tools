#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys

import numpy as np
import pandas as pd


def eval_CE_tl(df_gt_data, df_est, eval_timerange=[], right_on=False):
    """
    Calculate Circular-Error timeline

    Parameters
    ----------
    df_gt_data : pandas.DataFrame
        Ground truth, columns: [timestamp, x, y, theta, floor]
    df_est : pandas.DataFrame
        Estimated position, columns: [timestamp, x, y, floor, ...]
    eval_timerange : list
        Time range, [float, float]
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

    # Set eval timerange
    if type(eval_timerange) == str:
        df_timerange = pd.read_csv(eval_timerange, header=0)
    elif len(eval_timerange) < 1:
        df_timerange = pd.DataFrame(
            [[df_gt_data.index[0], df_gt_data.index[-1]]], columns=["ts_start", "ts_end"])
    elif is_2d_array(eval_timerange):
        df_timerange = pd.DataFrame(
            eval_timerange, columns=["ts_start", "ts_end"])
    else:
        df_timerange = pd.DataFrame([eval_timerange], columns=[
            "ts_start", "ts_end"])

    if right_on:
        df_eval = pd.merge_asof(df_est, df_gt_data,
                                left_index=True, right_index=True, tolerance=0.5, direction='nearest',
                                suffixes=["_est", "_gt"])
    else:
        df_eval = pd.merge_asof(df_gt_data, df_est,
                                left_index=True, right_index=True, tolerance=0.5, direction='nearest',
                                suffixes=["_gt", "_est"])
    df_eval = df_eval.dropna(subset=("x_gt", "y_gt", "x_est", "y_est"))

    print(len(df_eval))
    df_eval = filter_timerange(df_eval, df_timerange)
    print(len(df_eval))

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


def is_2d_array(lst):
    try:
        arr = np.array(lst)
        return arr.ndim == 2
    except:
        return False


def filter_timerange(df, df_timerange):
    if len(df_timerange) == 0:
        # 区間がなければ空の結果
        df_in = df.iloc[0:0].copy()
    else:
        # ベクトル化して「どれか1つの区間に入っている」かを判定（両端含む）
        ts = df["timestamp"].to_numpy()
        starts = df_timerange["ts_start"].to_numpy()
        ends = df_timerange["ts_end"].to_numpy()

        # shape: (len(df), len(df_VISO))
        in_any = (ts[:, None] >= starts[None, :]) & (
            ts[:, None] <= ends[None, :])

        # 行方向に「どれか True があるか」
        mask = in_any.any(axis=1)

        df_in = df.loc[mask].copy()
    return df_in


def eval_CE(df_est, step, quantile=50):
    """
    Entry point for integrated localization in step format
    """
    if 'gt' not in step.keys():
        raise KeyError(F'[gt.data] is not found')
    gt = step['gt']['data']

    df_ce_tl = eval_CE_tl(gt, df_est)

    return np.percentile(df_ce_tl.value, quantile)
