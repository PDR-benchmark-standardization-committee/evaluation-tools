#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import argparse

import numpy as np

from evaltools.com_tools.frfw_tools import *


def main(eval_middle_filename_ble):
    """
    BLE評価結果の出力
    BLEビーコン位置推定系のみ対応

    Parameters
    ----------
    eval_middle_filename_ble : str or DataFrame
        Format : [bdaddress, type, value]

    Returns
    ----------
    result_dict : dictionary
    """
    df_mb = load_csv(eval_middle_filename_ble)

    type_list = np.unique(df_mb.type)
    result_dict = {}
    for type_tag in type_list:
        df_type = df_mb[df_mb.type == type_tag]

        res = get_evalresult_bdaddress(df_type)

        result_dict[F'{type_tag}'] = res

    return result_dict


def get_evalresult_bdaddress(df_type):
    result = {}
    result['avg'] = df_type.value.mean()
    result['median'] = df_type.value.median()
    result['min'] = df_type.value.min()
    result['max'] = df_type.value.max()
    result['per50'] = np.percentile(df_type.value, 50)
    result['per75'] = np.percentile(df_type.value, 75)
    result['per90'] = np.percentile(df_type.value, 90)

    return result


def main_cl():
    parser = argparse.ArgumentParser()

    parser.add_argument('-eval_middle_file', '-m')

    args = parser.parse_args()

    result_dict = main(args.eval_middle_file)

    print(result_dict)


if __name__ == '__main__':
    main_cl()
