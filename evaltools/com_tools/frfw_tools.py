#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import json

import pandas as pd


def load_csv(filename):
    """
    text(filename) -> pd.read_csv
    other -> return
    """
    if type(filename) == str:
        return pd.read_csv(filename, index_col=0)

    return filename


def load_pickle(filename):
    """
    text(filename) -> pickle.load
    other -> return
    """
    if type(filename) == str:
        return pd.read_pickle(filename)

    return filename


def load_json(filename):
    """
    text(filename) -> json.load
    other -> return
    """
    if type(filename) == str:
        with open(filename, 'r') as fr:
            json_data = json.load(fr)
        return json_data

    return filename


def get_filename_info(filename, tag):
    """
    1_2_est.csv
     => (1_2, 2)
    """
    filename = filename.split(os.sep)[-1]
    filename = filename.split('.')[0]  # 拡張子
    if tag != '':
        dataname = filename.split(F'{tag}')[0]  # _est, _gt

    id = dataname.split('_')[-1]
    return (dataname, id)
