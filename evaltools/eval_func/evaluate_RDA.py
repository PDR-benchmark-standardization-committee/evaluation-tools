#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

debug = False


def evaluate_RDA_tl(est1, est2, gt1, gt2, setname=''):
    """
    Calculate Relative-Distance-Accuracy Error

    Computes the per-timestamp Euclidean distances between two estimated trajectories,
    and the error with respect to the corresponding ground-truth distances.

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
    setname : String
        Tag used to separate results by type.
        Deprecated: use an additional column to distinguish result types instead.

    Returns
    -------
    result: pandas.DataFrame
        Error at each timestamp, columns: [timestamp, type, value]
    """
    # 時刻同期
    df1 = pd.merge_asof(est1, gt1, on='timestamp', tolerance=0.1,
                        direction='nearest', suffixes=['_est1', '_gt1'])
    df2 = pd.merge_asof(est2, gt2, on='timestamp', tolerance=0.1,
                        direction='nearest', suffixes=['_est2', '_gt2'])
    df = pd.merge_asof(df1, df2, on='timestamp', tolerance=0.1,
                       direction='nearest').set_index('timestamp')

    df.dropna(subset=["x_est1", "x_est2", "x_gt1", "x_gt2", "y_est1", "y_est2",
              "y_gt1", "y_gt2"], inplace=True)

    RDA = np.sqrt((df.x_est1 - df.x_est2)**2 + (df.y_est1 - df.y_est2)**2) - \
        np.sqrt((df.x_gt1 - df.x_gt2)**2 + (df.y_gt1 - df.y_gt2)**2)

    if debug:
        df['est_distance'] = np.sqrt(
            (df.x_est1 - df.x_est2)**2 + (df.y_est1 - df.y_est2)**2)
        df['gt_distace'] = np.sqrt(
            (df.x_gt1 - df.x_gt2)**2 + (df.y_gt1 - df.y_gt2)**2)
        df['RDA'] = np.abs(RDA)
        df.to_csv('debug_RHA.csv')

    type_text = F'rda_{setname}' if setname != '' else 'rda'
    df_rda_tl = pd.DataFrame(data={'timestamp': df.index,
                                   'type': type_text, 'value': np.abs(RDA)}).set_index('timestamp')

    return df_rda_tl


def plot_graph(RDA, est1, dataname):
    # Plot RDA
    plt.figure(figsize=(10, 6))
    plt.plot(est1['timestamp'], RDA,
             label='Relative Distance Accuracy (RDA)', color='blue')
    plt.xlabel('Timestamp')
    plt.ylabel('RDA')
    plt.title('Relative Distance Accuracy (RDA) Over Time')
    plt.legend()
    plt.grid(True)

    # plt.show()
    plt.savefig(F'RDA_{dataname}.png', dpi=200)
    plt.close()


def main_cl(args):
    est1 = pd.read_csv(F'{args.est1}', header=0)
    est2 = pd.read_csv(F'{args.est2}', header=0)
    gt1 = pd.read_csv(F'{args.gt1}', header=0)
    gt2 = pd.read_csv(F'{args.gt2}', header=0)

    RDA = evaluate_RDA_tl(est1, est2, gt1, gt2)

    # plot_graph(RDA, est1, args.dataname)

    return RDA


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-est1')
    parser.add_argument('-est2')
    parser.add_argument('-gt1')
    parser.add_argument('-gt2')

    parser.add_argument('--dataname', default='00')

    args = parser.parse_args()
    main_cl(args)
