#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys
import pandas as pd
import numpy as np
import pickle
import copy
from scipy.optimize import minimize
from scipy.optimize import basinhopping
from scipy.special import huber
import matplotlib.pyplot as plt

###
# 推定軌跡からbleビーコン位置を推定
# ideal_rssi = -10 * FSPL * np.log10(dist) + TxPower
# --------------------------------------------------------------
def est_beacon_pos(bdaddress, df_est, blescans, df_merged = None, thresh = -85, txpower = -70, loss_rate = 2.1, initial_guess=[0, 0],
                   method = 'MSE', r = 1):
    """
    BLE logと位置座標情報からBLEビーコン位置を再投影する。
    """
    blescans_bdaddress = blescans[blescans.bdaddress == bdaddress]
    if df_merged is None:
        if blescans_bdaddress.index.name != 'ts': blescans_bdaddress.index.name = 'ts'
        if df_est.index.name != 'ts': df_est.index.name = 'ts'
        df_merged = pd.merge_asof(blescans_bdaddress, df_est, on='ts', direction='nearest', tolerance=0.05)
    else:
        df_merged = copy.deepcopy(df_merged)
        df_merged['x'] = df_merged['x_est']
        df_merged['y'] = df_merged['y_est']
    df_merged = df_merged[df_merged.rssi > thresh]

    if len(df_merged) < 1:
        # print('Empty df_merged')
        return (None, None, None)
    
    df_merged["z"] = 1.2

    # BLEビーコンの位置推定関数
    def estimate_position(coords, txpower, loss_rate):
        # distance = ((df_merged['x'] - coords[0])**2 + (df_merged['y'] - coords[1])**2 + (df_merged['z'] - coords[2])**2)**0.5
        distance = ((df_merged['x'] - coords[0])**2 + (df_merged['y'] - coords[1])**2 + (df_merged['z'] - 1.2)**2)**0.5
        predicted_rssi = estimate_rssi(distance, txpower, loss_rate)
        
        # print(predicted_rssi.values, df_merged['rssi'].values)
        # print(method, r)
        error = np.sum(loss_function(predicted_rssi.values, df_merged['rssi'].values, method, r))
        # print(error)
        return error
    
    # 最適化の実行
    constraints = ({'type': 'ineq', 'fun': lambda x: x[1]})
    result = minimize(estimate_position, initial_guess, method='SLSQP', constraints=constraints, args=(txpower, loss_rate))
    # result = basinhopping(estimate_position, initial_guess, niter=100, minimizer_kwargs={'args':(txpower,loss_rate), 'method':'SLSQP', 'constraints':constraints})
    xb, yb = result.x
    zb = 1.2
    return (xb, yb, zb)


def do_ble_estimation(df_est, blescans, gis_ble, thresh=-85, txpower=-70, loss_rate=2.1, method='MSE', r=1):
    """
    各BLEビーコンに対する再投影位置座標と推定に使用したデータ量をデータフレームにして返す。
    """
    est_beacon_list = []
    for bdaddress, beacon in gis_ble.iterrows():
        x, y, z = est_beacon_pos(bdaddress, df_est, blescans, None, thresh, txpower=txpower, loss_rate=loss_rate, initial_guess=[beacon.x, beacon.y], method=method, r=r)
        if x is None: continue

        blescans_bdaddress = blescans[blescans.bdaddress == bdaddress]
        est_beacon_list.append([bdaddress, x, y, z, len(blescans_bdaddress[blescans_bdaddress.rssi > thresh])])
    
    est_beacon = np.array(est_beacon_list, dtype=object)
    
    df_ble_est = pd.DataFrame(est_beacon[:,1:], index=est_beacon[:,0], columns=['x', 'y', 'z', 'count'])
    df_ble_est.index.name = 'bdaddress'
    return df_ble_est


#-------------------------------------------------------------------------------------------------------
# フリスの伝達式を用いたRSSI推定関数
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



if __name__ == '__main__':
    pass
