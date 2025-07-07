#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys
import argparse
import pandas as pd
import numpy as np
import math

from pfloc_optimization.igoz.evtools import do_ble_estimation


def eval_re(df_est, step, method, R, thresh, deltaT):
    """
    統合測位用の(step形式の)呼び出し口
    """
    if 'blescans' not in step.keys(): raise KeyError(F'[blescans] is not found')
    blescans = step['blescans']
    if 'gis' not in step.keys() or 'ble' not in step['gis'].keys(): raise KeyError(F'[gis.ble] is not found')
    gis_ble = step['gis']['ble']
    
    return eval_RE(df_est, blescans, gis_ble, method=method, R=R, deltaT=deltaT)


def calc_RE(df_est, blescans, gis_ble, thresh=-85, txpower=-70, loss_rate=2.1, method='MSE', R=1, output='avg'):
    if blescans.index.name != 'ts': blescans.index.name = 'ts'
    if df_est.index.name != 'ts': df_est.index.name = 'ts'
    df_merged = pd.merge_asof(blescans, df_est, on='ts', direction='nearest', tolerance=0.05)
    df_merged = pd.merge(df_merged, gis_ble, on='bdaddress', suffixes=['_est', '_ble'])
    df_merged = df_merged.dropna(subset=('x_est', 'y_est'))
    df_merged = df_merged[df_merged.rssi > thresh]

    distance = ((df_merged.x_est - df_merged.x_ble)**2 + (df_merged.y_est - df_merged.y_ble)**2 + (1.2 - df_merged.height)**2)**(1/2)
    predicted_rssi = do_ble_estimation.estimate_rssi(distance, txpower, loss_rate)

    errors = do_ble_estimation.loss_function(predicted_rssi, df_merged.rssi, method, R)

    if pd.isnull(errors).any():
        print((df_merged.x_est - df_merged.x_ble))
        print((df_merged.y_est - df_merged.y_ble))
        print((1.2 - df_merged.height))

        #print(df_est)
        #print(blescans)
        sys.exit()

    if output == 'avg': return np.sum(errors)/len(errors)
    elif output == 'per50': return np.percentile(errors, 50)
    elif output == 'diff_median':
        return np.percentile(np.abs(predicted_rssi-df_merged.rssi), 50)
    else: return errors 


def eval_RE(df_est, blescans, gis_ble, thresh=-65, txpower=-70, loss_rate=2.1, method='MSE', R=1, deltaT=10):
    s = math.floor(df_est.index[0])
    e = math.ceil(df_est.index[-1])

    re = []
    for ts_start in range(s, e):
        delta_pfloc = df_est[ts_start:ts_start+deltaT]
        delta_blescans = blescans[ts_start:ts_start+deltaT]

        if len(delta_pfloc) < 1 or len(delta_blescans[delta_blescans.rssi > thresh]) < 1: continue

        delta_RE = calc_RE(delta_pfloc, delta_blescans, gis_ble,
                               txpower=txpower, loss_rate=loss_rate,
                               output='diff_median')
        re.append(delta_RE)
    
    errors = do_ble_estimation.loss_function(re, np.zeros(len(re)), method, R)

    return np.sum(errors)/len(errors)


def main_cl(args):
    eval_re()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    main_cl(args)
