#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from joblib import Parallel, delayed


def eval_EAG_tl(df_gt_data, df_est, ALIP_timerange=[], mode='T', is_realtime=False, verbose=False):
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
    # Set ALIP timerange
    if type(ALIP_timerange) == str:
        df_ALIP = pd.read_csv(ALIP_timerange, header=0).astype(float)
    elif len(ALIP_timerange) < 1:
        df_ALIP = pd.DataFrame(
            [[df_gt_data.index[0], df_gt_data.index[-1]]], columns=["ts_start", "ts_end"])
    elif is_2d_array(ALIP_timerange):
        df_ALIP = pd.DataFrame(ALIP_timerange, columns=["ts_start", "ts_end"])
    else:
        df_ALIP = pd.DataFrame([ALIP_timerange], columns=[
                               "ts_start", "ts_end"])

    # Execute in parallel
    result_dfs = Parallel(n_jobs=-1, backend="loky", verbose=0)(
        delayed(process_row)(df_gt_data, df_est, [
            row.ts_start, row.ts_end], mode, is_realtime, verbose)
        for row in df_ALIP.itertuples(index=False)
    )

    # Combine results
    if len(result_dfs) < 1:
        return None
    df_eag_tl = pd.concat(result_dfs)
    df_eag_tl.sort_values("timestamp", inplace=True)

    return df_eag_tl


def process_row(df_gt_data, df_est, ALIP_timerange, mode, is_realtime, verbose):
    try:
        df_eag_tl_ALIP = calc_EAG(
            df_gt_data, df_est, ALIP_timerange, is_realtime)
    except Exception as e:
        print(e)

    return df_eag_tl_ALIP


def is_2d_array(lst):
    try:
        arr = np.array(lst)
        return arr.ndim == 2
    except:
        return False


###
def calc_T_EAG(df_gt_data, df_est, ALIP_timerange=[], is_realtime=False, verbose=False):
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

    df_gt_data['delta_ts'] = df_gt_data.index
    mid = ALIP_start + (ALIP_end - ALIP_start) / 2

    # Restrict data to the ALIP timerange
    if is_realtime:
        df_gt_data['delta_ts'] -= ALIP_start
    else:
        df_gt_data.loc[(df_gt_data.index >= ALIP_start) & (
            df_gt_data.index < mid), 'delta_ts'] -= ALIP_start
        df_gt_data.loc[(df_gt_data.index >= mid) & (
            df_gt_data.index <= ALIP_end), 'delta_ts'] -= ALIP_end
        df_gt_data.loc[(df_gt_data.index >= mid) & (
            df_gt_data.index <= ALIP_end), 'delta_ts'] *= -1

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


def calc_EAG(df_gt_data, df_est, ALIP_timerange=[], is_realtime=False):
    """
    Returns
    -------
    df_eag_tl : pandas.DataFrame
        Error at each timestamp, columns: [timestamp, type, value]
        type : eag
        value : [error_distance_from_gt];[ALIP_elapsed_time];[ALIP_elapsed_distance];[ALIP_elapsed_angle]
    """
    # set timerange
    if len(ALIP_timerange) < 1:
        ALIP_timerange = (df_gt_data.index[0], df_gt_data.index[-1])
    ALIP_start = ALIP_timerange[0]
    ALIP_end = ALIP_timerange[-1]

    df_gt_data = set_index_timestamp(df_gt_data)
    df_est = set_index_timestamp(df_est)

    df_gt_data = df_gt_data.dropna(subset=("x", "y"))
    df_est = df_est.dropna(subset=("x", "y"))

    df_gt_data['ts_gt'] = df_gt_data.index
    df_est['ts_est'] = df_est.index

    # check yaw
    if "yaw" not in df_gt_data.keys() and "qx" in df_gt_data.keys():
        df_gt_data["yaw"] = [Rotation.from_quat(q).as_euler(
            "XYZ")[0] for q in df_gt_data[["qx", "qy", "qz", "qw"]].values]
    if "yaw" not in df_est.keys() and "qx" in df_est.keys():
        df_gt_data["yaw"] = [Rotation.from_quat(q).as_euler(
            "XYZ")[0] for q in df_est[["qx", "qy", "qz", "qw"]].values]

    df_eval = pd.merge_asof(df_gt_data, df_est,
                            left_index=True, right_index=True, tolerance=0.5, direction='nearest',
                            suffixes=["_gt", "_est"])
    df_eval = df_eval.dropna(subset=("x_gt", "y_gt", "x_est", "y_est"))

    # floor check
    if "floor_est" in df_eval.columns and "floor_gt" in df_eval.columns:
        df_eval["floor_correct"] = (
            df_eval["floor_est"] == df_eval["floor_gt"])
        df_eval = df_eval[df_eval['floor_correct']]
    df_eval = df_eval.loc[ALIP_start+0.001:ALIP_end-0.001]

    if len(df_eval) < 1:
        return pd.DataFrame(columns=["timestamp", "type", "value"]).set_index('timestamp')

    #
    error_from_gt = np.hypot(
        df_eval['x_gt'] - df_eval['x_est'], df_eval['y_gt'] - df_eval['y_est'])
    elapsed_time = calc_elapsed_time(
        df_eval.copy(deep=True), ALIP_start, ALIP_end, is_realtime)
    elapsed_distance = calc_elapsed_distance(
        df_eval.copy(deep=True), ALIP_start, ALIP_end)
    elapsed_angle = calc_elapsed_angle(
        df_eval.copy(deep=True), ALIP_start, ALIP_end)

    value_arr = combine_arrays_with_semicolon(
        error_from_gt, elapsed_time, elapsed_distance, elapsed_angle)
    df_eag_tl = pd.DataFrame(
        data={'timestamp': df_eval.index, 'type': 'eag', 'value': value_arr}).set_index('timestamp')

    return df_eag_tl


def set_index_timestamp(df):
    """
    """
    if isinstance(df.index, pd.RangeIndex):
        if "timestamp" in df.columns:
            df.set_index("timestamp", inplace=True)
        elif "unixtime" in df.columns:
            df.set_index("unixtime", inplace=True)

    return df


def calc_elapsed_time(df_eval, ALIP_start, ALIP_end, is_realtime):
    """
    """
    df_eval['delta_ts'] = df_eval.index
    mid = ALIP_start + (ALIP_end - ALIP_start) / 2

    # Restrict data to the ALIP timerange
    if is_realtime:
        df_eval['delta_ts'] -= ALIP_start
    else:
        df_eval.loc[(df_eval.index >= ALIP_start) & (
            df_eval.index < mid), 'delta_ts'] -= ALIP_start
        df_eval.loc[(df_eval.index >= mid) & (
            df_eval.index <= ALIP_end), 'delta_ts'] -= ALIP_end
        df_eval.loc[(df_eval.index >= mid) & (
            df_eval.index <= ALIP_end), 'delta_ts'] *= -1

    return df_eval['delta_ts'].values


def calc_elapsed_distance(df_eval, ALIP_start, ALIP_end):
    """
    """
    df_eval['x_diff'] = df_eval['x_est'].diff()
    df_eval['y_diff'] = df_eval['y_est'].diff()
    # df_eval.drop(index=df_eval.index[0], inplace=True)
    df_eval['x_diff'].fillna(0, inplace=True)
    df_eval['y_diff'].fillna(0, inplace=True)
    df_eval['dist_diff'] = np.hypot(
        df_eval['x_diff'], df_eval['y_diff'])

    def calc_Sdistance(row):
        idx = row['ts_gt']

        dist_s = np.sum(df_eval.loc[ALIP_start:idx]['dist_diff'].values)
        dist_e = np.sum(df_eval.loc[idx:ALIP_end]['dist_diff'].values)

        return np.min([dist_s, dist_e])

    dist_from_ALIPpoint = df_eval.apply(calc_Sdistance, axis=1)
    return dist_from_ALIPpoint


def calc_elapsed_angle(df_eval, ALIP_start, ALIP_end):
    """
    Notes
    -----
    If yaw is missing, return an np.array filled with NaNs
    """
    if "yaw_gt" not in df_eval.keys() or "yaw_est" not in df_eval.keys():
        return np.full(len(df_eval), np.nan)

    df_eval['yaw_gt'] = np.abs(df_eval['yaw_gt'].diff())
    df_eval['yaw_est'] = np.abs(df_eval['yaw_est'].diff())
    # df_eval.drop(index=df_eval.index[0], inplace=True)
    df_eval.fillna(0, inplace=True)

    def calc_Srad(row):
        idx = row['ts_gt']
        Srad_s = np.sum(df_eval.loc[ALIP_start:idx]['yaw_est'].values)
        Srad_e = np.sum(df_eval.loc[idx:ALIP_end]['yaw_est'].values)

        return np.min([Srad_s, Srad_e])

    Srad_from_ALIP = df_eval.apply(calc_Srad, axis=1)
    return Srad_from_ALIP


def combine_arrays_with_semicolon(A, B, C, D):
    """
    Concatenate the error values for each time series into a semicolon-separated string.
    value : "{error_from_gt};{elapsed_time};{elapsed_distance};{elapsed_angle}"

    Parameters
    ----------
    A : pandas.DataFrame
        error_from_gt at each time
    B : pandas.DataFrame
        elapsed_time at each time
    C : pandas.DataFrame
        elapsed_distance at each time
    D : pandas.DataFrame
        elapsed_angle at each time

    Returns
    -------
    combined : pandas.DataFrame
        [timestamp(index), "eag", "A;B;C;D"]
    """
    A, B, C, D = map(np.asarray, (A, B, C, D))
    if not (A.shape == B.shape == C.shape == D.shape):
        print(A, B, C, D)
        raise ValueError("All input arrays must have the same shape")

    A_s = np.asarray(A, dtype=str)
    B_s = np.asarray(B, dtype=str)
    C_s = np.asarray(C, dtype=str)
    D_s = np.asarray(D, dtype=str)

    combined = np.char.add(
        np.char.add(
            np.char.add(A_s, np.char.add(';', B_s)),
            np.char.add(';', C_s)
        ),
        np.char.add(';', D_s)
    )
    return combined


def eval_EAG(df_est, step):
    """
    Entry point for integrated localization in step format
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
