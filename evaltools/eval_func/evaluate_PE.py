#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys
import argparse

import pandas as pd
import numpy as np
import copy


def eval_pe(df_est, step, target_rate):
     """
    統合測位用の(step形式の)呼び出し口
    """
     if 'xdr' not in step.keys(): raise KeyError(F'[xdr] is not found')
     pdr = step['xdr']

     return eval_PE(df_est, pdr, target_rate)


def eval_PE(pfloc, pdr, target_rate = 1.3, valid_vel = 1.2):
    """
    pdrの移動距離に対してpfloc移動距離が一定倍率を超えないかチェック
    """
    df_pfloc = calc_velocity_from_xy2(copy.deepcopy(pfloc))
    df_pdr = calc_velocity_from_xy2(copy.deepcopy(pdr))
    
    def calc_route(row):
        if row['vel'] > valid_vel: return 0
        return row['ddist']
    
    tmp = df_pfloc.apply(calc_route, axis=1)
    pfloc_length = np.sum(tmp)

    tmp = df_pdr.apply(calc_route, axis=1)
    pdr_length = np.sum(tmp)

    # return (pfloc_length/pdr_length, pfloc_length, pdr_length)
    return np.abs(pfloc_length/pdr_length - target_rate)


def calc_velocity_from_xy2(df_est, x_column='x', y_column='y', window_time = 0.5):
    """
    Calc velocity from sum of move distance from -0.5sec to +0.5sec
    """
    if 'ts' not in df_est.columns: df_est['ts'] = df_est.index
    
    dx = df_est[x_column].diff(); dx.iloc[0] = 0
    dy = df_est[y_column].diff(); dy.iloc[0] = 0
    ddist = np.sqrt(dx.values**2 + dy.values**2)

    dt = df_est['ts'].diff(); dt.iloc[0] = 1 # avoiding 0-division error
    df_vel = pd.DataFrame(data={'vel':ddist/dt, 'ddist':ddist, 'ts':df_est['ts'].values}, index=df_est['ts'].values)

    def calc_velocity_window(row):
        df_target = df_vel[row['ts'] - window_time:row['ts'] + window_time]
        return np.average(df_target['vel'].values)
    
    df_vel['vel'] = df_vel.apply(calc_velocity_window, axis=1)

    return df_vel


def main_cl(args):
    steps = pd.read_pickle('')
    PE, pfc, pdr = eval_pe(steps['pfloc'], steps['xdr'])
    print(PE, pfc, pdr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    main_cl(args)
