#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys

import numpy as np
import pandas as pd


def eval_VE_tl(df_est, df_gt=None, window_time=0.5):
    """
    Calc Velocity Error

    Prameters
    ---------
    df_est : pandas.DataFrame
        Estimated position, columns: [timestamp, x, y, floor, ...]
    df_gt : pandas.DataFrame
            Ground-truth, columns: [timestamp, x, y, theta, floor]
    window_time : float
        interval time of calcurating velocity
        vel(t) = {f(t+window_time) - f(t-window_time)}/window_time

    Returns
    -------
    result: pandas.DataFrame
        Error at each timestamp, columns: [timestamp, type, value]
    """
    if df_gt is not None:
        df_ve_tl = calc_velocity_error_with_gt(df_est, df_gt, window_time)

    else:
        df_ve_tl = calc_velocity_error_negative_xy(df_est)

    return df_ve_tl


def calc_velocity_error_with_gt(df_est, df_gt, window_time=0.5):
    """
    Calculate error of (gt_vel - est_vel)
    """
    df_est = df_est.dropna(subset=('x', 'y'))

    # merge with timestamp
    df_eval = pd.merge_asof(df_gt, df_est,
                            left_index=True, right_index=True, tolerance=0.5, direction='nearest',
                            suffixes=["_gt", "_est"])
    df_eval = df_eval.dropna(subset=("x_gt", "y_gt", "x_est", "y_est"))

    # calc velocity
    est_velocity = calc_velocity_from_xy2(
        df_eval, 'x_est', 'y_est', window_time)
    gt_velocity = calc_velocity_from_xy2(df_eval, 'x_gt', 'y_gt', window_time)

    error_tl = np.abs(gt_velocity - est_velocity)
    df_ve_tl = pd.DataFrame(data={'timestamp': df_eval.index,
                            'type': 've_gt', 'value': error_tl}).set_index('timestamp')

    return df_ve_tl


def calc_velocity_error_negative_xy(df_est, window_time=0.5):
    """
    check abnormal walking velocity (= valid_vel)
    and calculate rating of abnormal velocity.

    calculate velocity from x-y version
    """
    df_est = df_est.dropna(subset=('x', 'y'))

    # calc velocity from x-y
    velocity = calc_velocity_from_xy2(df_est, 'x', 'y', window_time)
    df_ve_tl = pd.DataFrame(data={'timestamp': df_est.index,
                            'type': 've_neg', 'value': velocity}).set_index('timestamp')

    return df_ve_tl


def calc_velocity_from_xy2(df_est, x_column='x', y_column='y', window_time=0.5):
    """
    Calc velocity from sum of move distance from -0.5sec to +0.5sec
    """
    if 'ts' not in df_est.columns:
        df_est['ts'] = df_est.index

    dx = df_est[x_column].diff()
    dx.iloc[0] = 0
    dy = df_est[y_column].diff()
    dy.iloc[0] = 0
    ddist = np.sqrt(dx.values**2 + dy.values**2)

    dt = df_est['ts'].diff()
    dt.iloc[0] = 1  # avoiding 0-division error
    df_vel = pd.DataFrame(
        data={'vel': ddist/dt, 'ts': df_est['ts'].values}, index=df_est['ts'].values)

    def calc_velocity_window(row):
        df_target = df_vel[row['ts'] - window_time:row['ts'] + window_time]
        return np.average(df_target['vel'].values)

    velocity_list = df_vel.apply(calc_velocity_window, axis=1)

    return velocity_list


def eval_VE(df_est, step=None, valid_vel=1.5):
    """
    統合測位用の(step形式の)呼び出し口
    """
    if step is not None:
        if 'gt' not in step.keys():
            raise KeyError(F'[gt.data] is not found')
        gt = step['gt']['data']
    else:
        gt = None

    df_ve_tl = eval_VE_tl(df_est, df_gt=gt)

    if gt is not None:
        return np.percentile(df_ve_tl.value, 50)
    else:
        # calc abnormaly rate
        err_vel = df_ve_tl.value[df_ve_tl.value > valid_vel]
        return len(err_vel)/len(df_ve_tl.value)


# ----------------------------------------------------------------------
def main_negative_vxvy(df_est, valid_vel=1.5, est_timerange=[]):
    """
    check abnormal walking velocity (= valid_vel)
    and calculate rate of abnormal velocity.

    calculate velocity from vx-vy version
    """
    df_est = df_est.dropna(subset=("vx", "vy"))

    # calc velocity from vx-vy
    velocity = np.sqrt(df_est['vx']**2 + df_est['vy']**2)

    # set timerange
    if len(est_timerange) > 0:
        velocity = velocity[str(est_timerange[0]):str(est_timerange[1])]

    # calc rate
    err_vel = velocity[velocity > valid_vel]
    return round(np.sum(err_vel)/len(df_est), 4)


def calc_velocity_from_xy(df_est, x_column='x', y_column='y', diff_rate=1):
    """
    Calc velocity from x-y
    """
    if 'ts' not in df_est.columns:
        df_est['ts'] = df_est.index

    dx = df_est[x_column].diff(diff_rate)
    dx = dx.iloc[diff_rate:]
    dy = df_est[y_column].diff(diff_rate)
    dy = dy.iloc[diff_rate:]
    dt = df_est['ts'].diff(diff_rate)
    dt = dt.iloc[diff_rate:]

    return np.sqrt(dx.values**2 + dy.values**2)/dt.values
