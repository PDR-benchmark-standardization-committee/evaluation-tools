#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys
import numpy as np
import pandas as pd
import copy
from scipy.spatial.transform import Rotation

def eval_pdr_rad(gt_data, xdr, squareError=True):
    if squareError: return eval_RadError2(gt_data, xdr)
    else:
        return eval_RadError(gt_data, xdr)


def eval_RadError(gt_data, xdr):
    """
    総角度変化量
    """
    gt_start = gt_data.index[0]
    gt_end = gt_data.index[-1]

    xdr_data = xdr[gt_start:gt_end]

    if 'yaw' not in gt_data.keys():
        gt_data['yaw'] = [Rotation.from_quat(q).as_euler("XYZ")[0] for q in gt_data[["q0", "q1", "q2", "q3"]].values]
    gt_yaw = gt_data.yaw
    gt_yaw -= gt_yaw.iloc[0]
    gt_unwrap_yaw = np.unwrap(gt_yaw)

    xdr_yaw = xdr_data.yaw
    xdr_yaw -= xdr_yaw.iloc[0]
    xdr_unwrap_yaw = np.unwrap(xdr_yaw)

    return abs(gt_unwrap_yaw[-1] + xdr_unwrap_yaw[-1])


def eval_RadError2(gt_data, xdr):
    """
    各時刻方位角の二乗誤差
    """
    gt_start = gt_data.index[0]
    gt_end = gt_data.index[-1]

    xdr_data = xdr[gt_start:gt_end]

    if 'yaw' not in gt_data.keys():
        gt_data['yaw'] = [Rotation.from_quat(q).as_euler("XYZ")[0] for q in gt_data[["q0", "q1", "q2", "q3"]].values]
    
    gt_yaw = gt_data.yaw
    gt_yaw -= gt_yaw.iloc[0]
    gt_unwrap_yaw = np.unwrap(gt_yaw)

    xdr_yaw = xdr_data.yaw
    xdr_yaw -= xdr_yaw.iloc[0]
    xdr_unwrap_yaw = np.unwrap(xdr_yaw)

    gt_df = pd.DataFrame(data=gt_unwrap_yaw, index=gt_data.index)
    xdr_df = pd.DataFrame(data=xdr_unwrap_yaw, index=xdr_data.index)
    if xdr_df.index.name != 'ts': xdr_df['ts'] = xdr_df.index
    if gt_df.index.name != 'ts': gt_df['ts'] = gt_df.index
    df = pd.merge_asof(xdr_df, gt_df, on='ts', direction='nearest', tolerance=0.1)
    df['0_y'] -= df['0_y'].iloc[0]
    print(df)
    tmp = (df['0_x'] + df['0_y']) ** 2
    
    return np.sum(tmp)
