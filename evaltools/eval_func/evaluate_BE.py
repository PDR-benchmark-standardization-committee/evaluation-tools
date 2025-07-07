#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys
import argparse

import pandas as pd
import numpy as np

from pfloc_optimization.igoz.evtools import do_ble_estimation


def eval_be(df_est, step, method, R):
    """
    統合測位用の(step形式の)呼び出し口
    """
    if 'blescans' not in step.keys(): raise KeyError(F'[blescans] is not found')
    blescans = step['blescans']
    if 'gis' not in step.keys() or 'ble' not in step['gis'].keys(): raise KeyError(F'[gis.ble] is not found')
    gis_ble = step['gis']['ble']

    return eval_BE(df_est, blescans, gis_ble, method=method, R=R)[0]


def eval_BE(df_est, blescans, gis_ble, thresh=-85, txpower=-70, loss_rate=2.1, method='MSE', R=1):
    df_ble_est = do_ble_estimation.do_ble_estimation(df_est, blescans, gis_ble, thresh, txpower, loss_rate, method, R)
    
    df_eval = pd.merge(df_ble_est, gis_ble, on='bdaddress', suffixes=['_est', '_ble'], how='left')
    
    def calc_hypot(row, blescans):
        return np.hypot(row['x_est'] - row['x_ble'], row['y_est'] - row['y_ble']) # * (row['count']/len(blescans))
    
    df_eval['BE'] = df_eval.apply(calc_hypot, blescans=blescans, axis=1)
    BE50 = np.percentile(df_eval['BE'].values, 50)

    return (BE50, df_eval)
    

def main_cl(args):
    eval_be()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    main_cl(args)
