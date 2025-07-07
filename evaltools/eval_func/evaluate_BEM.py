#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys
import argparse
import glob
import pandas as pd
import numpy as np
import math
import pickle
import copy
from scipy.optimize import minimize
from scipy.optimize import basinhopping


def eval_bem(df_est, step, nEstVar):
    """
    統合測位用の(step形式の)呼び出し口
    """
    if 'blescans' not in step.keys(): raise KeyError(F'[blescans] is not found')
    blescans = step['blescans']
    if 'gis' not in step.keys() or 'ble' not in step['gis'].keys(): raise KeyError(F'[gis.ble] is not found')
    gis_ble = step['gis']['ble']

    return eval_BEM(df_est, blescans, gis_ble, nEstVar=nEstVar)


def eval_BEM(df_est, blescans, gis_ble, txpower=-70, loss_rate=2.1, nEstVar=3, method='MSE', r=1):
    floor = np.unique(df_est['floor'])[0]
    if blescans.index.name != 'ts': blescans.index.name = 'ts'
    if df_est.index.name != 'ts': df_est.index.name = 'ts'
    df_merged = pd.merge_asof(blescans, df_est, on='ts', direction='nearest', tolerance=0.05)
    df_merged.dropna(subset=['x', 'y'], inplace=True) # Noneが混じるとエラーは出ないが推定座標が全て同じになる

    gis_ble_floor = gis_ble[gis_ble.floorname == floor]
    
    # BLEビーコンの位置推定関数
    def estimate_position(coords, gis_ble, txpower, loss_rate, O=(0,0)):
        gis_ble_floor[['X_rot', 'Y_rot']] = gis_ble.apply(rotation_mat, args=(coords[0], coords[1], coords[2], O), axis=1)

        if nEstVar==3: errors = gis_ble_floor.apply(calc_rssi_error, args=(df_merged, txpower, loss_rate, method, r), axis=1)   # 3変数
        elif nEstVar==4: errors = gis_ble.apply(calc_rssi_error, args=(df_merged, coords[3], loss_rate, method, r), axis=1) # 4変数
        elif nEstVar==5: errors = gis_ble.apply(calc_rssi_error, args=(df_merged, coords[3], coords[4], method, r), axis=1) # 5変数
        errors = [e for e in errors if not np.isnan(e)]

        return np.sum(errors)/len(errors)
    

    # 最適化の実行
    constraints = ({'type': 'ineq', 'fun': lambda x: x[2]})
    Xw = np.average(gis_ble_floor.x); Yw = np.average(gis_ble_floor.y)

    initial_guess = [Xw,Yw,0.0]
    if nEstVar>=4: initial_guess.append(-70)
    if nEstVar>=5: initial_guess.append(2.0)
    
    result = minimize(estimate_position, initial_guess, method='SLSQP', constraints=constraints, args=(gis_ble_floor, txpower, loss_rate, (Xw, Yw)))
    # result = basinhopping(estimate_position, initial_guess, niter=100, minimizer_kwargs={'args':(gis_ble, txpower,loss_rate), 'method':'SLSQP', 'constraints':constraints})

    gis_ble_floor[['X_rot','Y_rot']] = gis_ble_floor.apply(rotation_mat, args=(result.x[0], result.x[1], result.x[2], (Xw, Yw)), axis=1)
    bem = gis_ble_floor.apply(calc_beacon_error, axis=1)

    return np.sum(bem)/len(bem)


def rotation_mat(row, x0, y0, theta, O=(0,0)):
    X = (row.x + x0) * np.cos(theta) - (row.y + y0) * np.sin(theta)
    Y = (row.x + x0) * np.sin(theta) + (row.y + y0) * np.cos(theta)

    return pd.Series([X - O[0], Y - O[1]])


def calc_rssi_error(row, df_merged, txpower, loss_rate, method, r):
    df_merged = df_merged[df_merged.bdaddress == row.name]
    if len(df_merged) < 1: return None

    distance = ((df_merged.x - row.X_rot)**2 + (df_merged.y - row.Y_rot)**2)**0.5
    predicted_rssi = estimate_rssi(distance, txpower, loss_rate)

    return np.sum(loss_function(predicted_rssi.values, df_merged['rssi'].values, method, r))/len(predicted_rssi)


def estimate_rssi(distance, txpower=-70, loss_rate=2.1):
    return txpower - (10 * loss_rate *  np.log10(distance))


# 損失関数群
def loss_function(predicted_rssi, actual_rssi, method, r=1):
    if method == 'MSE' or method == 'mse':
        # Mean Squared Error
        return (predicted_rssi - actual_rssi) **2
    
    # elif method == 'Huber' or method == 'huber':
    #     # Huber loss
    #     return huber(np.abs(predicted_rssi - actual_rssi), r)

    elif method == 'Huber' or method == 'huber':
        # Huber loss (handmade)
        tmp = []
        for pr, ar in zip(predicted_rssi, actual_rssi):
            if np.abs(pr - ar) <= r:
                res = (predicted_rssi - actual_rssi)**2/2
            else:
                res = r*np.abs(predicted_rssi - actual_rssi) - (r**2/2)
            tmp.append(res)
        
        return tmp
    
    elif method == 'Cauchy' or method == 'cauchy':
        # Cauchy loss
        return ((r**2)/2) * np.log(1+((predicted_rssi - actual_rssi)/r)**2)


def calc_beacon_error(row):
    return ((row.x - row.X_rot)**2 + (row.y - row.Y_rot)**2)**0.5

