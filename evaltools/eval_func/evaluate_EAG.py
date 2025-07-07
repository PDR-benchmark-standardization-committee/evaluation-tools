#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation


def eval_EAG_tl(df_gt_data, df_est, ALIP_timerange=[], mode='T', verbose=False):
    """
    Calculate Error Accumulation Gradient

    Parameters
    ----------
    df_gt_data : pandas.DataFrame
        Ground truth, columns: [timestamp, x, y, theta, floor]
    df_est : pandas.DataFrame
        Estimated position, columns: [timestamp, x, y, floor, ...]
    ALIP_timerange : list of float
        Time range, [float, float]
    mode : {'T', 'D', 'A'}
        Mode of EAG calculation: Time, Distance, or Angle.
    verbose : boolean
        Added intermediate data used for graph plotting to the return value.

    Returns
    -------
    result : pandas.DataFrame
        Error at each timestamp, columns: [timestamp, type, value]
    """
    try:
        if mode == 'T':
            df_eag_tl = calc_T_EAG(df_gt_data, df_est, ALIP_timerange, verbose)
        elif mode == 'D':
            df_eag_tl = calc_D_EAG(df_gt_data, df_est, ALIP_timerange)
        elif mode == 'A':
            df_eag_tl = calc_A_EAG(df_gt_data, df_est, ALIP_timerange)
    except Exception as e:
        print(e)

    return df_eag_tl


###
def calc_T_EAG(df_gt_data, df_est, ALIP_timerange=[], verbose=False):
    """
    Calculate EAG based on Time
    """
    # set timerange
    if len(ALIP_timerange) < 1:
        ALIP_timerange = (df_gt_data.index[0], df_gt_data.index[-1])
    ALIP_start = ALIP_timerange[0]
    ALIP_end = ALIP_timerange[-1]

    df_gt_data = df_gt_data[ALIP_start:ALIP_end]
    df_est = df_est[ALIP_start:ALIP_end]

    # convert to relative time near ALIP-start time and ALIP-end time
    df_gt_data['delta_ts'] = df_gt_data.index
    df_gt_data['delta_ts'][ALIP_start:ALIP_start +
                           (ALIP_end - ALIP_start)/2] -= ALIP_start
    df_gt_data['delta_ts'][ALIP_start +
                           (ALIP_end - ALIP_start)/2:ALIP_end] -= ALIP_end
    df_gt_data['delta_ts'][ALIP_start +
                           (ALIP_end - ALIP_start)/2:ALIP_end] *= -1

    df_gt_data = df_gt_data.dropna(subset=("x", "y"))
    df_est = df_est.dropna(subset=("x", "y"))

    df_gt_data['ts_gt'] = df_gt_data.index
    df_est['ts_est'] = df_est.index

    # merge with timestamp
    df_eval = pd.merge_asof(df_gt_data, df_est,
                            left_index=True, right_index=True, tolerance=0.5, direction='nearest',
                            suffixes=["_gt", "_est"])
    df_eval = df_eval.dropna(subset=("x_gt", "y_gt", "x_est", "y_est"))

    # floor check
    if "floor_est" in df_eval.columns and "floor_gt" in df_eval.columns:
        df_eval["floor_correct"] = (
            df_eval["floor_est"] == df_eval["floor_gt"])
        df_eval = df_eval[df_eval['floor_correct']]
    df_eval = df_eval[ALIP_start+0.001:ALIP_end-0.001]

    error_from_gt = np.hypot(
        df_eval['x_gt'] - df_eval['x_est'], df_eval['y_gt'] - df_eval['y_est'])

    df_EAG = error_from_gt / df_eval['delta_ts']

    df_eag_tl = pd.DataFrame(
        data={'timestamp': df_eval.index, 'type': 'eag', 'value': df_EAG}).set_index('timestamp')

    if verbose:
        return df_eag_tl, error_from_gt, df_eval['delta_ts']
    else:
        return df_eag_tl

    # ### graph
    # mask = (df_eval_FC['delta_ts'] > 1)

    # fig = plt.figure()
    # ax1 = fig.add_subplot(2,1,1)
    # ax1.set_xlabel('elapsed time [s]')
    # ax1.set_ylabel('error distance form gt [m]')
    # ax1.scatter(df_eval_FC['delta_ts'], np.hypot(df_eval_FC['x_gt'] - df_eval_FC['x_est'], df_eval_FC['y_gt'] - df_eval_FC['y_est']), s=1)

    # ax2 = fig.add_subplot(2,1,2)
    # ax2.scatter(df_eval_FC['delta_ts'][mask], df_EAG[mask], s=1)
    # ax2.set_xlabel('elapsed time [s]')
    # ax2.set_ylabel('T-EAG [m/s]')

    # p50 = np.full(len(df_eval_FC['delta_ts'][mask]), np.percentile(df_EAG.values, 50))

    # plt.tight_layout()
    # fig.savefig(output_path + 'T_EAG.png')
    # plt.close()

###


def calc_D_EAG(df_gt_data, df_est, ALIP_timerange=[], draw_flg=False, output_path='./output/'):
    """
    Calculate EAG based on Distance
    """
    if len(ALIP_timerange) < 1:
        ALIP_timerange = (df_gt_data.index[0], df_gt_data.index[-1])
    ALIP_start = ALIP_timerange[0]
    ALIP_end = ALIP_timerange[-1]

    df_gt_data = df_gt_data[ALIP_start:ALIP_end]
    df_est = df_est[ALIP_start:ALIP_end]

    s_point = (df_gt_data['x'].iloc[0], df_gt_data['y'].iloc[0])
    e_point = (df_gt_data['x'].iloc[-1], df_gt_data['y'].iloc[-1])

    # if "floor_ble_mode" in df_est.columns:
    #     df_est["floor"] = df_est["floor_ble_mode"]
    df_gt_data = df_gt_data.dropna(subset=("x", "y"))
    df_est = df_est.dropna(subset=("x", "y"))

    df_gt_data['ts_gt'] = df_gt_data.index
    df_est['ts_est'] = df_est.index

    df_eval = pd.merge_asof(df_gt_data, df_est,
                            left_index=True, right_index=True, tolerance=0.5, direction='nearest',
                            suffixes=["_gt", "_est"])
    df_eval = df_eval.dropna(subset=("x_gt", "y_gt", "x_est", "y_est"))

    df_eval["floor_correct"] = (df_eval["floor_est"] == df_eval["floor_gt"])
    df_eval_FC = df_eval[df_eval['floor_correct']]
    df_eval_FC = df_eval_FC[ALIP_start+0.001:ALIP_end-0.001]

    df_eval_FC['x_diff'] = df_eval_FC['x_est'].diff()
    df_eval_FC['y_diff'] = df_eval_FC['y_est'].diff()
    df_eval_FC.drop(index=df_eval_FC.index[0], inplace=True)
    df_eval_FC['dist_diff'] = np.hypot(
        df_eval_FC['x_diff'], df_eval_FC['y_diff'])

    def calc_Sdistance(row):
        idx = row['ts_gt']
        dist_s = np.sum(df_eval_FC[ALIP_start:idx]['dist_diff'].values)
        dist_e = np.sum(df_eval_FC[idx:ALIP_end]['dist_diff'].values)

        return np.min([dist_s, dist_e])

    dist_from_ALIPpoint = df_eval_FC.apply(calc_Sdistance, axis=1)
    error_from_gt = np.hypot(
        df_eval_FC['x_gt'] - df_eval_FC['x_est'], df_eval_FC['y_gt'] - df_eval_FC['y_est'])

    df_EAG = error_from_gt/dist_from_ALIPpoint

    df_eag_tl = pd.DataFrame(
        data={'timestamp': df_eval_FC.index, 'type': 'eag', 'value': df_EAG}).set_index('timestamp')

    return df_eag_tl

###


def calc_A_EAG(df_gt_data, df_est, ALIP_timerange=[], draw_flg=False, output_path='./output/'):
    """
    Calculate EAG based on Angle
    """
    if len(ALIP_timerange) < 1:
        ALIP_timerange = (df_gt_data.index[0], df_gt_data.index[-1])
    ALIP_start = ALIP_timerange[0]
    ALIP_end = ALIP_timerange[-1]

    df_gt_data = df_gt_data[ALIP_start:ALIP_end]
    df_est = df_est[ALIP_start:ALIP_end]

    if "yaw" not in df_gt_data.keys():
        df_gt_data["yaw"] = [Rotation.from_quat(q).as_euler(
            "XYZ")[0] for q in df_gt_data[["q0", "q1", "q2", "q3"]].values]

    # if "floor_ble_mode" in df_est.columns:
    #     df_est["floor"] = df_est["floor_ble_mode"]
    df_gt_data = df_gt_data.dropna(subset=("x", "y"))
    df_est = df_est.dropna(subset=("x", "y"))

    df_gt_data['ts_gt'] = df_gt_data.index
    df_est['ts_est'] = df_est.index

    df_eval = pd.merge_asof(df_gt_data, df_est,
                            left_index=True, right_index=True, tolerance=0.5, direction='nearest',
                            suffixes=["_gt", "_est"])
    df_eval = df_eval.dropna(subset=("x_gt", "y_gt", "x_est", "y_est"))

    df_eval["floor_correct"] = (df_eval["floor_est"] == df_eval["floor_gt"])
    df_eval_FC = df_eval[df_eval['floor_correct']]
    df_eval_FC = df_eval_FC[ALIP_start+0.001:ALIP_end-0.001]

    df_eval_FC['yaw_gt'] = np.abs(df_eval_FC['yaw_gt'].diff())
    df_eval_FC['yaw_est'] = np.abs(df_eval_FC['yaw_est'].diff())
    df_eval_FC.drop(index=df_eval_FC.index[0], inplace=True)

    def calc_Srad(row):
        idx = row['ts_gt']
        Srad_s = np.sum(df_eval_FC[ALIP_start:idx]['yaw_est'].values)
        Srad_e = np.sum(df_eval_FC[idx:ALIP_end]['yaw_est'].values)

        return np.min([Srad_s, Srad_e])

    Srad_from_ALIP = df_eval_FC.apply(calc_Srad, axis=1)
    error_from_gt = np.hypot(
        df_eval_FC['x_gt'] - df_eval_FC['x_est'], df_eval_FC['y_gt'] - df_eval_FC['y_est'])

    df_EAG = error_from_gt/Srad_from_ALIP

    df_eag_tl = pd.DataFrame(
        data={'timestamp': df_eval_FC.index, 'type': 'eag', 'value': df_EAG}).set_index('timestamp')

    return df_eag_tl


def eval_EAG(df_est, step):
    """
    統合測位用の(step形式の)呼び出し口
    """
    if 'gt' not in step.keys():
        raise KeyError(F'[gt.data] is not found')
    gt = step['gt']['data']

    df_eag_tl = eval_EAG_tl(gt, df_est)

    return np.percentile(df_eag_tl.value, 50)


def draw_eag(error_from_gt, Srad_from_ALIP, mode, output_path='./'):
    df_EAG = error_from_gt/Srad_from_ALIP

    # graph
    mask = (Srad_from_ALIP > 1)
    if mode == 'T':
        x_label_text = 'elapsed time [s]'
        # ax1.scatter(df_eval_FC['delta_ts'], np.hypot(df_eval_FC['x_gt'] - df_eval_FC['x_est'], df_eval_FC['y_gt'] - df_eval_FC['y_est']), s=1)
    elif mode == 'D':
        x_label_text = 'cumlative distance [m]'
    elif mode == 'A':
        x_label_text = 'cumlative angle [rad]'

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.set_xlabel(x_label_text)
    ax1.set_ylabel('error distance form gt [m]')
    ax1.scatter(Srad_from_ALIP, error_from_gt, s=1)

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.set_xlabel('elapsed time [s]')
    ax2.set_ylabel(F'{mode}-EAG')
    ax2.scatter(Srad_from_ALIP[mask], df_EAG[mask], s=1)
    plt.tight_layout()
    fig.savefig(output_path + F'{mode}_EAG.png')
    plt.close()
