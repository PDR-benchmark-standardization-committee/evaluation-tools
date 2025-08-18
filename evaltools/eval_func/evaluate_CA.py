#!/usr/bin/env python
# -*- coding: utf-8 -*-
from evaltools.com_tools.convert_tools import convert_quat_to_yaw, normalize_rad
from scipy.spatial.transform import Rotation
import seaborn as sns
import scipy.stats as kde
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
import math
import warnings
warnings.simplefilter('ignore')


def eval_CA_tl(df_gt_data, df_est, mode_CS='RCS'):
    """
    Calcurate Coordinate-Accuracy-Error timeline

    Parameters
    ----------
    df_gt_data : pandas.DataFrame
        Ground truth, columns: [timestamp, x, y, theta, floor]
    df_est : pandas.DataFrame
        Estimated position, columns: [timestamp, x, y, floor, ...]
    mode_CS : {'RCS', 'ACS'}
        set x-y Coordinate System 'yaw-Relative' or 'Absolute'

    Returns
    ----------
    df_ca_tl : pandas.DataFrame
        Error at each timestamp, columns: [timestamp, type, value]
        type: 'ca_x', 'ca_y'
    """
    df_eval = calc_xy_errors(df_est, df_gt_data, mode_CS)

    df_cax_tl = pd.DataFrame(
        data={'timestamp': df_eval.index, 'type': F'ca_x_{mode_CS}', 'value': df_eval['x_error']}).set_index('timestamp')
    df_cay_tl = pd.DataFrame(
        data={'timestamp': df_eval.index, 'type': F'ca_y_{mode_CS}', 'value': df_eval['y_error']}).set_index('timestamp')
    df_ca_tl = pd.concat([df_cax_tl, df_cay_tl])

    return df_ca_tl


def calc_xy_errors(df_est, df_gt_data, mode='RCS'):
    """
    """
    # yaw check
    df_est = convert_quat_to_yaw(df_est)
    df_gt_data = convert_quat_to_yaw(df_gt_data)
    # if "yaw" not in df_gt_data.keys():
    #     df_gt_data["yaw"] = [Rotation.from_quat(q).as_euler(
    #         "XYZ")[0] for q in df_gt_data[["q0", "q1", "q2", "q3"]].values]
    # if "yaw" not in df_est.keys():
    #     df_est["yaw"] = calc_yaw_from_xy(df_est)

    # floor check
    if "floor_ble_mode" in df_est.columns:
        df_est["floor"] = df_est["floor_ble_mode"]

    # merge
    df_gt_data = df_gt_data.dropna(subset=("x", "y"))
    df_est = df_est.dropna(subset=("x", "y"))
    df_eval = pd.merge_asof(df_gt_data, df_est,
                            left_index=True, right_index=True, tolerance=0.5, direction='nearest',
                            suffixes=["_gt", "_est"])
    df_eval = df_eval.dropna(subset=("x_gt", "y_gt", "x_est", "y_est"))

    # floor correct
    if "floor_est" in df_eval.columns and "floor_gt" in df_eval.columns:
        df_eval["floor_correct"] = (
            df_eval["floor_est"] == df_eval["floor_gt"])
        df_eval = df_eval[df_eval['floor_correct']]

    # calc (x_error, y_error)
    if mode == 'RCS':
        # Relative Coordinate System
        df_eval['y_error'] = df_eval.apply(calc_y_error, axis=1)
        df_eval['x_error'] = df_eval.apply(calc_x_error, axis=1)
    elif mode == 'ACS':
        # Absolute Coordinate System
        df_eval['x_error'] = df_eval['x_gt'] - df_eval['x_est']
        df_eval['y_error'] = df_eval['y_gt'] - df_eval['y_est']

    return df_eval


def calc_y_error(row):
    """
    orthographic vector
    y_error = {(vector_a ・ vector_b) / norm(ventor_a)**2}|vector_a|

    vector_a : unit vector to yaw direction
    vector_b : vector from estimated position to ground truth position
    y_error = (vector_a ・ vector_b)
    """
    vector_b = [row['x_gt'] - row['x_est'], row['y_gt'] - row['y_est']]
    vector_a = [np.sin(row['yaw_est']), np.cos(row['yaw_est'])]

    return (np.dot(vector_a, vector_b))


def calc_x_error(row):
    """
    vector_b : vector from estimated position to ground truth position
    vector_c : unit vector normal to yaw direction
    x_error = (vector_c ・ vector_b)
    """
    vector_b = [row['x_gt'] - row['x_est'], row['y_gt'] - row['y_est']]
    vector_c = [np.cos(row['yaw_est']), -np.sin(row['yaw_est'])]

    return (np.dot(vector_c, vector_b))


def calc_yaw_from_xy(df_est):
    """
    +y = 0; -y = 2pi
    +y → +x → -y = (0 <= Θ < 2pi)
    +y → -x → -y = (-2pi < Θ <= 0)
    """
    return np.arctan(df_est['x']/df_est['y'])


# ----------------------------------------------------------------------
def eval_CA(df_est, step, mode_CS='RCS', mode_density='kde'):
    """
    Entry point for integrated localization in step format
    """
    if 'gt' not in step.keys():
        raise KeyError(F'[gt.data] is not found')
    gt = step['gt']['data']

    df_ca_tl = eval_CA_tl(gt, df_est, mode_CS)
    x_error = df_ca_tl[df_ca_tl.type == 'ca_x'].value
    y_error = df_ca_tl[df_ca_tl.type == 'ca_y'].value

    if mode_density == 'kde':
        CA = calc_kernel_mod(x_error, y_error)
    elif mode_density == '2dh':
        CA = calc_2dhist_mod(x_error, y_error)

    if np.isnan(CA):
        raise ValueError('CA is NAN')
    else:
        return CA


# Kernel Density Estimation
def calc_kernel_density_mod(x, y, output_path, RSC_flg, bw_method=None):
    fig = plt.figure()
    sns.set_style('whitegrid')
    plt.rcParams['font.size'] = 12
    nbins = 300
    k = kde.gaussian_kde([x, y], bw_method=bw_method)
    xi, yi = np.mgrid[min(x)-2:max(x)+2:nbins*1j, min(y)-2:max(y)+2:nbins*1j]
    try:
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    except:
        print('Unable to calculate inverse matrix, return mean value')
        return np.mean(x), np.mean(y), fig
    row_idx = np.argmax(zi) // len(xi)
    col_idx = np.argmax(zi) % len(yi)
    x_mod = xi[:, 0][row_idx].round(2)
    y_mod = yi[0][col_idx].round(2)

    plt.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap='jet')
    plt.plot(x_mod, y_mod, marker='^', color='forestgreen',
             markerfacecolor='white', markeredgewidth=2, markersize=12)
    plt.title('x: {:.2f} y: {:.2f}'.format(x_mod, y_mod))
    plt.xlabel('X error')
    plt.ylabel('Y error')

    sc = 'RCS' if RSC_flg else 'ACS'
    fig.savefig(output_path + F'CA_kernel_{sc}.png')
    plt.close()

    CA = math.hypot(x_mod, y_mod)

    return CA


def calc_kernel_mod(x, y, bw_method=None):
    nbins = 300
    k = kde.gaussian_kde([x, y], bw_method=bw_method)
    xi, yi = np.mgrid[min(x)-2:max(x)+2:nbins*1j, min(y)-2:max(y)+2:nbins*1j]
    try:
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    except:
        print('Unable to calculate inverse matrix, return mean value')
        return np.mean(x), np.mean(y)
    row_idx = np.argmax(zi) // len(xi)
    col_idx = np.argmax(zi) % len(yi)
    x_mod = xi[:, 0][row_idx].round(2)
    y_mod = yi[0][col_idx].round(2)

    CA = math.hypot(x_mod, y_mod)

    return CA


# 2d Histgram Estimation
def calc_2D_histgram_mod(x_error_list, y_error_list):

    fig = plt.figure()

    plt.rcParams['font.size'] = 12
    xmax = max(np.abs(x_error_list))
    ymax = max(np.abs(y_error_list))

    xbin = math.floor(xmax * 2/0.5)
    ybin = math.floor(ymax * 2/0.5)

    counts, xedges, yedges, _ = plt.hist2d(
        x_error_list, y_error_list, bins=(xbin, ybin))
    x_delta = xedges[1] - xedges[0]
    y_delta = yedges[1] - yedges[0]

    idx = np.unravel_index(np.argmax(counts), counts.shape)

    x_mod = xedges[idx[0]] + x_delta/2
    y_mod = yedges[idx[1]] + y_delta/2

    plt.plot(x_mod, y_mod, marker='^', color='forestgreen',
             markerfacecolor='white', markeredgewidth=2, markersize=12)
    plt.xlabel('X error')
    plt.ylabel('Y error')
    plt.close()

    fig.savefig('CA_2dhist.png')

    CA = math.hypot(x_mod, y_mod)

    return CA


def calc_2dhist_mod(x_error_list, y_error_list):
    """
    Generate distributions of the position error in the x and y components separately,  
    compute the mean of each distribution, and return the distance from the origin (== Ground-truth) as the evaluation value.
    """
    xmax = max(np.abs(x_error_list))
    ymax = max(np.abs(y_error_list))

    xbin = math.floor(xmax * 2/0.5)
    ybin = math.floor(ymax * 2/0.5)

    counts, xedges, yedges, _ = plt.hist2d(
        x_error_list, y_error_list, bins=(xbin, ybin))
    x_delta = xedges[1] - xedges[0]
    y_delta = yedges[1] - yedges[0]

    # print(np.argmax(counts))
    idx = np.unravel_index(np.argmax(counts), counts.shape)

    x_mod = xedges[idx[0]] + x_delta/2
    y_mod = yedges[idx[1]] + y_delta/2

    x_mod, y_mod

    CA = math.hypot(x_mod, y_mod)

    return CA
