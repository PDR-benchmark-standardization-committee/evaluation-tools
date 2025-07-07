#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from evaltools.com_tools.convert_tools import convert_quot_to_yaw

debug = False


def evaluate_RHA_tl(est1, est2, gt1, gt2, setname=''):
    """
    RelativeHeadingAccuracy
    2軌跡間の時刻毎方位角差とその正解軌跡間の時刻毎方位角差の誤差
    """
    # check yaw
    est1 = convert_quot_to_yaw(est1)
    est2 = convert_quot_to_yaw(est2)
    gt1 = convert_quot_to_yaw(gt1)
    gt2 = convert_quot_to_yaw(gt2)

    # 時刻同期
    df1 = pd.merge_asof(est1, gt1, on='timestamp', tolerance=0.1,
                        direction='nearest', suffixes=['_est1', '_gt1'])
    df2 = pd.merge_asof(est2, gt2, on='timestamp', tolerance=0.1,
                        direction='nearest', suffixes=['_est2', '_gt2'])
    df = pd.merge_asof(df1, df2, on='timestamp', tolerance=0.1,
                       direction='nearest').set_index('timestamp')

    df.dropna(subset=["yaw_est1", "yaw_est2",
              "yaw_gt1", "yaw_gt2"], inplace=True)

    # _RHA = ((df.yaw_est1 - df.yaw_est2).apply(scale_rad) -
    #        (df.yaw_gt1 - df.yaw_gt2).apply(scale_rad)).apply(scale_rad)
    RHA = ((df.yaw_est1 - df.yaw_est2) -
           (df.yaw_gt1 - df.yaw_gt2)).apply(scale_rad)

    if debug:
        df['est_disyaw'] = (df.yaw_est1 - df.yaw_est2).apply(scale_rad)
        df['gt_disyaw'] = (df.yaw_gt1 - df.yaw_gt2).apply(scale_rad)
        df['RHA'] = np.abs(RHA)
        df.to_csv('debug_RHA.csv')

        # print(np.sum(np.abs(RHA - _RHA)))

    type_text = F'rha_{setname}' if setname != '' else 'rha'
    df_rha_tl = pd.DataFrame(data={'timestamp': df.index,
                                   'type': type_text, 'value': np.abs(RHA)}).set_index('timestamp')

    return df_rha_tl


def scale_rad(rad):
    """
    -pi < θ <= pi へスケーリング
    """
    if -1*np.pi < rad <= np.pi:
        return rad
    elif rad <= -1 * np.pi:
        rad += 2*np.pi
        return scale_rad(rad)
    elif np.pi < rad:
        rad -= 2*np.pi
        return scale_rad(rad)
    else:
        print(F'RHA : {rad}')
        return None


def plot_graph(RHA, est1, dataname):
    # Plot RDA
    plt.figure(figsize=(10, 6))
    plt.plot(est1['timestamp'], RHA,
             label='Relative Heading Accuracy (RHA)', color='blue')
    plt.xlabel('Timestamp')
    plt.ylabel('RHA')
    plt.title('Relative Heading Accuracy (RHA) Over Time')
    plt.legend()
    plt.grid(True)

    # plt.show()
    plt.savefig(F'RHA_{dataname}.png', dpi=200)
    plt.close()


def main_cl(args):
    est1 = pd.read_csv(F'{args.est1}', header=0)
    est2 = pd.read_csv(F'{args.est2}', header=0)
    gt1 = pd.read_csv(F'{args.gt1}', header=0)
    gt2 = pd.read_csv(F'{args.gt2}', header=0)

    RDA = evaluate_RHA_tl(est1, est2, gt1, gt2)

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
