#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import argparse

import pandas as pd

from evaltools.com_tools.frfw_tools import *


def main(pickle_filename, target_traj):
    """
    """
    pic_dict = load_pickle(pickle_filename)

    if target_traj == 'gt':
        target_data = pic_dict[target_data]['data']
    else:
        target_data = pic_dict[target_traj]

    df_target = check_format(target_data)
    if type(pickle_filename) == str:
        fname = pickle_filename.split('.pickle')[0].split(os.sep)[-1]
    else:
        fname = 'target_traj'
    df_target.to_csv(fname + '.csv')


def check_format(target_data):
    timestamp = target_data.index
    x = target_data.x
    y = target_data.y
    yaw = target_data.yaw
    if 'floor_ble_mode' in target_data.keys():
        floor = target_data.floor_ble_mode
    else:
        floor = target_data.floor

    df = pd.DataFrame(data={'timestamp': timestamp, 'x': x,
                            'y': y, 'yaw': yaw, 'floor': floor})
    df.set_index('timestamp')
    del df['timestamp']
    return df


def main_cl():
    parser = argparse.ArgumentParser()

    parser.add_argument('-pickle', '-p')
    parser.add_argument('-target', '-t')

    args = parser.parse_args()

    main(args.pickle, args.target)


if __name__ == '__main__':

    main_cl()
