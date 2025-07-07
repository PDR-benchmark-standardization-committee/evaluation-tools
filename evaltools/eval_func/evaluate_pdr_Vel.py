#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys
import numpy as np
import pandas as pd
import copy

def eval_pdr_vel(gt_data, xdr):
    return eval_VelError2(gt_data, xdr)


def eval_VelError(steps, xdr):
    """
    速度誤差
    """
    gt_data = steps['gt']['data']
    gt_start = gt_data.index[0]
    gt_end = gt_data.index[-1]

    xdr_data = xdr[gt_start:gt_end]

    xdr_vel = np.sqrt(xdr_data.vx ** 2 + xdr_data.vy ** 2)
    diff_period = 100
    steps["gt"]["data"]["ts"] = steps["gt"]["data"].index.values
    gt_vel = np.sqrt(gt_data.x.diff(diff_period) ** 2 \
                     + gt_data.y.diff(diff_period) ** 2) \
                     / gt_data.ts.diff(diff_period)
    gt_vel = pd.DataFrame(gt_vel.dropna())

    tmp = []
    for index, vel in gt_vel.iterrows():
        target_index = xdr_vel.index.get_loc(index, method='nearest')
        tmp.append((vel - xdr_vel.iloc[target_index])**2)
    
    return np.sum(tmp)


def eval_VelError2(gt_data, xdr, column_x='vx', column_y='vy'):
    """
    平滑化後速度誤差評価
    """
    gt_start = gt_data.index[0]
    gt_end = gt_data.index[-1]

    xdr_data = xdr[gt_start:gt_end]

    xdr_vel = np.sqrt(xdr_data[column_x] ** 2 + xdr_data[column_y] ** 2)
    diff_period = 100
    gt_data["ts"] = gt_data.index.values
    gt_vel = np.sqrt(gt_data.x.diff(diff_period) ** 2 \
                     + gt_data.y.diff(diff_period) ** 2) \
                     / gt_data.ts.diff(diff_period)
    gt_vel = gt_vel.dropna(); gt_index = np.array(gt_vel.index)

    kernel = np.ones(201)/201
    xdr_vel = np.convolve(xdr_vel, kernel, mode='same')
    xdr_vel = pd.DataFrame(data=xdr_vel, index=xdr_data.index)
    gt_vel = np.convolve(gt_vel.values, kernel, mode='same')
    gt_vel = pd.DataFrame(data=gt_vel, index=gt_index)

    xdr_vel['ts'] = xdr_vel.index
    gt_vel['ts'] = gt_vel.index
    df = pd.merge_asof(xdr_vel, gt_vel, on='ts', direction='nearest', tolerance=0.1)
    
    tmp = (df['0_x'] - df['0_y']) ** 2
    
    return np.sum(tmp)


def eval_SumDist(gt_data, xdr):
    """
    xdrとgtの総移動距離の誤差を評価
    """
    gt_start = gt_data.index[0]
    gt_end = gt_data.index[-1]

    xdr_data = xdr[gt_start:gt_end]

    SumDist_gt = calc_SumDist(gt_data)
    SumDist_xdr = calc_SumDist(xdr_data)

    return abs(SumDist_gt - SumDist_xdr)


def calc_SumDist(data):
    X = data['x'].values
    Y = data['y'].values
    
    return np.sum(np.sqrt(np.square(np.diff(X)) + np.square(np.diff(Y))))


