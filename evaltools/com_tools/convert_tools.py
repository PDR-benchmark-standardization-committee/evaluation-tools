#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import math

import pandas as pd
from scipy.spatial.transform import Rotation as R


def convert_quot_to_yaw(df):
    # check yaw
    if 'yaw' not in df.columns:
        if 'qx' not in df.columns:
            df.rename(columns={'q1': 'qx', 'q2': 'qy',
                               'q3': 'qz', 'q4': 'qw'}, inplace=True)
        df['yaw'] = df.apply(quat_to_yaw, axis=1)

    return df


def quat_to_yaw(row):
    return math.atan2(2.0 * (row.qw * row.qz + row.qx * row.qy), 1.0 - 2.0 * (row.qy**2 + row.qz**2))


def calc_axis_from_quat(q1, q2, q3, q4, return_axis=False):
    rotation = R.from_quat([q1, q2, q3, q4])
    axis = rotation.as_matrix()

    if return_axis:
        return (axis[:, 0], axis[:, 1], axis[:, 2])
    else:
        return axis


def normalize_rad(rad, upper=False):
    """
    default:
        -pi <= rad < pi へ正規化

    upper -> True:
        -pi < rad <= pi へ正規化
    """
    if upper:
        return (rad - math.pi) % (2 * math.pi) - math.pi
    else:
        return (rad + math.pi) % (2 * math.pi) - math.pi
