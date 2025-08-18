#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import argparse
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from evaltools.com_tools.convert_tools import convert_quat_to_yaw, normalize_rad

debug = False


def evaluate_RPA_tl(est1, est2, gt1, gt2, id1=1, id2=2, expose_raw=False):
    """
    Calculate Relative-Pose-Accuracy Error

    Computes the relative polar coordinate error between two estimated trajectories,
    compared to the corresponding ground-truth relative polar coordinates.

    Parameters
    ----------
    est1 : pandas.DataFrame
        Estimated trajectory1, columns, [timestamp, x, y, (z,) yaw] or [timestamp, x, y, (z,) qx, qy, qz, qw]
    est2 : pandas.DataFrame
        Estimated trajectory2, columns, [timestamp, x, y, (z,) yaw] or [timestamp, x, y, (z,) qx, qy, qz, qw]
    gt1 : pandas.DataFrame
        Ground-truth trajectory corresponding to `est1`, columns, [timestamp, x, y, (z,) yaw] or [timestamp, x, y, (z,) qx, qy, qz, qw]
    gt2 : pandas.DataFrame
        Ground-truth trajectory corresponding to `est2`, columns, [timestamp, x, y, (z,) yaw] or [timestamp, x, y, (z,) qx, qy, qz, qw]
    id1 : String
        A label used to distinguish RPA values computed from the relative coordinates of est1 and est2.
    id2 : String
        A label used to distinguish RPA values computed from the relative coordinates of est1 and est2.

    Returns
    -------
    result: pandas.DataFrame
        Error at each timestamp, columns: [timestamp, type, value]
        type : [rpa_{id1}{id2}, rpa_{id2}{id1}]
    """
    est1 = convert_quat_to_yaw(est1)
    est2 = convert_quat_to_yaw(est2)
    gt1 = convert_quat_to_yaw(gt1)
    gt2 = convert_quat_to_yaw(gt2)

    df1 = pd.merge_asof(est1, gt1, on='timestamp', tolerance=0.1,
                        direction='nearest', suffixes=['_est1', '_gt1'])
    df2 = pd.merge_asof(est2, gt2, on='timestamp', tolerance=0.1,
                        direction='nearest', suffixes=['_est2', '_gt2'])
    df = pd.merge_asof(df1, df2, on='timestamp', tolerance=0.1,
                       direction='nearest').set_index('timestamp')

    df.dropna(subset=["yaw_est1", "yaw_est2", "yaw_gt1", "yaw_gt2", "x_est1", "x_est2",
              "x_gt1", "x_gt2", "y_est1", "y_est2", "y_gt1", "y_gt2"], inplace=True)

    vec12_yaw_est = np.arctan2(df.y_est2 - df.y_est1, df.x_est2 - df.x_est1)
    vec12_yaw_est_loc = (vec12_yaw_est - df.yaw_est1).apply(normalize_rad)
    vec21_yaw_est = np.arctan2(df.y_est1 - df.y_est2, df.x_est1 - df.x_est2)
    vec21_yaw_est_loc = (vec21_yaw_est - df.yaw_est2).apply(normalize_rad)
    vec12_yaw_gt = np.arctan2(df.y_gt2 - df.y_gt1, df.x_gt2 - df.x_gt1)
    vec12_yaw_gt_loc = (vec12_yaw_gt - df.yaw_gt1).apply(normalize_rad)
    vec21_yaw_gt = np.arctan2(df.y_gt1 - df.y_gt2, df.x_gt1 - df.x_gt2)
    vec21_yaw_gt_loc = (vec21_yaw_gt - df.yaw_gt2).apply(normalize_rad)

    RPA_12 = (vec12_yaw_est_loc - vec12_yaw_gt_loc).apply(normalize_rad)
    RPA_12 = np.abs(RPA_12)
    RPA_21 = (vec21_yaw_est_loc - vec21_yaw_gt_loc).apply(normalize_rad)
    RPA_21 = np.abs(RPA_21)

    df_rpa12_tl = pd.DataFrame(data={'timestamp': df.index,
                                     'type': F'rpa_{id1}{id2}', 'value': RPA_12}).set_index('timestamp')
    df_rpa21_tl = pd.DataFrame(data={'timestamp': df.index,
                                     'type': F'rpa_{id2}{id1}', 'value': RPA_21}).set_index('timestamp')

    df_rpa_tl = pd.concat([df_rpa12_tl, df_rpa21_tl])

    if expose_raw:
        return {
            'rpa': df_rpa_tl,
            'vec12_yaw_est': vec12_yaw_est,
            'vec12_yaw_est_loc': vec12_yaw_est_loc,
            'vec21_yaw_est': vec21_yaw_est,
            'vec21_yaw_est_loc': vec21_yaw_est_loc,
            'vec12_yaw_gt': vec12_yaw_gt,
            'vec12_yaw_gt_loc': vec12_yaw_gt_loc,
            'vec21_yaw_gt': vec21_yaw_gt,
            'vec21_yaw_gt_loc': vec21_yaw_gt_loc,
        }

    return df_rpa_tl


def main_cl(args):
    est1 = pd.read_csv(F'{args.est1}', header=0)
    est2 = pd.read_csv(F'{args.est2}', header=0)
    gt1 = pd.read_csv(F'{args.gt1}', header=0)
    gt2 = pd.read_csv(F'{args.gt2}', header=0)

    RPA = evaluate_RPA_tl(est1, est2, gt1, gt2)

    return RPA


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-est1')
    parser.add_argument('-est2')
    parser.add_argument('-gt1')
    parser.add_argument('-gt2')

    parser.add_argument('--dataname', default='00')

    args = parser.parse_args()
    main_cl(args)
