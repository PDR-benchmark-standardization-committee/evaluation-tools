#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys
import argparse
import glob
import pandas as pd
import numpy as np
import math
import itertools

from joblib import Parallel, delayed


def eval_dse(df_est, step, fillna, mode, n_jobs):
    """
    統合測位用の(step形式の)呼び出し口
    """
    if 'blescans' not in step.keys(): raise KeyError(F'[blescans] is not found')
    blescans = step['blescans']

    return eval_DSE(df_est, blescans, fillna, mode=mode, n_jobs=n_jobs)


def eval_DSE(df_est, blescans, fillna=-110, q=50, mode='R', n_jobs=30):
    """
    位置距離差に対するRssi値の類似性を評価
    """
    result = make_df(df_est, blescans, fillna, q, n_jobs)
    result = np.array(result)

    result_min = []
    for delta_dist in range(0, 10+1, 1):
        tmp = result[(delta_dist < result[:,0]) & (result[:,0] < delta_dist + 1)]
        if len(tmp) < 1: continue

        min_rssi_q = np.min(tmp[:,1])
        result_min.append([delta_dist, min_rssi_q])
    
    result_min = np.array(result_min)

    if mode == 'R': DSE = maximize_r2(result_min)
    elif mode == 'A': DSE = maximize_a(result_min)
    elif mode == 'RA':
        DSE = (maximize_r2(result_min) + maximize_a(result_min))
    else:
        DSE = 0
        print(mode)

    # if mode == 'min': DSE = minimize_residual(result_min)
    # elif mode == 'max': DSE = maximize_a(result_min)
    # else:
    #     DSE = None
    #     print(mode)

    return DSE


def make_df(df_est ,blescans, fillna=-110, q=50, n_jobs=5):
    res = resample_blescans_closest(blescans, df_est.index, internal_interval=0.01)
    if df_est.index.name != 'ts': df_est.index.name = 'ts'
    res = pd.merge_asof(res, df_est, on='ts', direction='nearest')
    res = pd.merge_asof(pd.DataFrame({"ts":[t/10 for t in range(0, int((res['ts'].values[-1]+1)*10), 5)]}), res, on='ts', direction='nearest')
    res.set_index("ts", inplace=True)
    
    dim_max = len(np.unique(blescans.bdaddress))
    
    result = itertools.combinations(res.index, 2)
    result = np.array_split([r for r in result], n_jobs)

    result_l = Parallel(n_jobs=n_jobs)(delayed(calc_similarity_2)(res, r_list, len(df_est.iloc[0]), fillna, q) for r_list in result)
    result_list = []
    for l in result_l:
        result_list.extend(l)

    # result_list = Parallel(n_jobs=-1, verbose=3)(delayed(calc_similarity)(res, r1, r2, len(df_est.iloc[0]), fillna) for r1, r2 in result)
    # result_list = [r for r in result_list if r is not None]
    

    # result_list = []
    # for r1, r2 in result:
    #     r = calc_similarity(res, r1, r2, len(df_est.iloc[0]))

    #     if r is not None: result_list.append(r)
    
    # result_list = np.array(result_list)
    
    return result_list


def resample_blescans_closest(blescans : pd.DataFrame, timestamps : np.ndarray, internal_interval=0.01, 
                              bdaddress_list=None, rssi_threshold=-200, tolerance=0.5, 
                              agg_method="mean", agg_window=1):
    if len(timestamps) == 0: return None
    
    if bdaddress_list is None:
        bdaddress_list = blescans.bdaddress.unique()
    
    bdaddress_list = np.unique(bdaddress_list)
    n_samples = np.ceil((timestamps[-1] - (timestamps[0] - agg_window*2))/internal_interval)
    ts_internal = np.arange(n_samples) * internal_interval + timestamps[0] - agg_window*2
    blescans_internal = pd.DataFrame({"ts":ts_internal})
    
    # allocate each RSSI to any of the gyro's timestamp.
    for bdaddress in bdaddress_list:
        df_tmp = blescans[blescans.bdaddress == bdaddress][["rssi"]].rename(columns={"rssi" : f"{bdaddress}"}) # extract information of one beacon
        blescans_internal = pd.merge_asof(blescans_internal, df_tmp, left_on="ts", right_index=True, tolerance=internal_interval, suffixes=["", ""])

    if agg_method is None and agg_window is not None:
        blescans_internal.set_index("ts", inplace=True)
        for column in blescans_internal.columns:
            blescans_internal[column] = blescans_internal[column].rolling(window=int(agg_window / internal_interval), center=True, min_periods=1).mean()
        blescans_internal.reset_index(inplace=True)
    elif agg_method == "mean" and agg_window is not None:
        blescans_internal.set_index("ts", inplace=True)
        for column in blescans_internal.columns:
            blescans_internal[column] = blescans_internal[column].rolling(window=int(agg_window / internal_interval), center=True, min_periods=1).mean()
        blescans_internal.reset_index(inplace=True)
    elif agg_method == "max" and agg_window is not None:
        blescans_internal.set_index("ts", inplace=True)
        for column in blescans_internal.columns:
            blescans_internal[column] = blescans_internal[column].rolling(window=int(agg_window / internal_interval), center=True, min_periods=1).max()
        blescans_internal.reset_index(inplace=True)
    blescans_r = pd.merge_asof(pd.DataFrame({"ts":timestamps}), blescans_internal, left_on="ts", right_on="ts", tolerance=tolerance, suffixes=["", ""])
    
    return blescans_r.set_index("ts").applymap(lambda x: np.nan if x < rssi_threshold else x).astype(pd.SparseDtype("float", np.nan))


def calc_similarity(res, r1, r2, cut_len=1, fillna=-110, q=50):
    tmp = pd.merge(res.loc[r1], res.loc[r2], left_index=True, right_index=True)
    tmp = tmp[0:-1 * cut_len]
    tmp = tmp.dropna(how='all')
    tmp = tmp.fillna(fillna)

    if len(tmp) < 1: return None
   
    # X1 = res.loc[r1].x; Y1 = res.loc[r1].y
    # X2 = res.loc[r2].x; Y2 = res.loc[r2].y
    dist = ((res.loc[r1].x - res.loc[r2].x)**2 + (res.loc[r1].y - res.loc[r2].y)**2)**(1/2)

    Drssi_r1 = estimate_distance_from_rssi(tmp[r1].values)
    Drssi_r2 = estimate_distance_from_rssi(tmp[r2].values)

    rssi_percentile = np.percentile(np.abs(Drssi_r1 - Drssi_r2), q)
    
    return [dist, rssi_percentile]


def calc_similarity_2(res, array, cut_len=1, fillna=-110, q=90):
    tmp_list = []
    for r1, r2 in array:
        rs = calc_similarity(res, r1, r2, cut_len, fillna, q)

        if rs is None: continue
        tmp_list.append(rs)
    
    return tmp_list


def estimate_rssi_from_distace(distance, txpower=-70, loss_rate=2.1):
    return txpower - (10 * loss_rate *  np.log10(distance))


def estimate_distance_from_rssi(rssi, txpower=-70, loss_rate=2.1):
    tmp = np.array(txpower - rssi, dtype=float)
    return 10**(tmp/(10 * loss_rate))


def minimize_residual(src):
    fit = np.polyfit(src[:,0], src[:,1], 1)
    predicted_func = np.poly1d(fit)
    predicted_x = predicted_func(src[:,0])

    return np.sum((src[:,1] - predicted_x)**2)/len(src)


def maximize_r2(src):
    fit = np.polyfit(src[:,0], src[:,1], 1)
    predicted_func = np.poly1d(fit)
    predicted_x = predicted_func(src[:,0])

    S_all = np.sum((src[:,1] - src[:,1].mean())**2)
    S_reg = np.sum((predicted_x - src[:,1].mean())**2)

    R2 = S_reg/S_all
    return R2


def maximize_a(src):
    fit = np.polyfit(src[:,0], src[:,1], 1)
    a = fit[0]

    return a

