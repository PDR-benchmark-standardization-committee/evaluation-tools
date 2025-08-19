#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from evaltools.com_tools.frfw_tools import *


def main(eval_middle_filenames, type_tag, output_dir, suffix, sections_filename=None):
    df_m = pd.concat([load_csv(filename)
                     for filename in eval_middle_filenames], ignore_index=True)

    # Set timerange
    if sections_filename is None:
        time_intervals = [[df_m.index.min(), df_m.index.max()]]
    else:
        time_intervals = load_json(sections_filename)['sections']
        if len(time_intervals) < 1:
            time_intervals = [[df_m.index.min(), df_m.index.max()]]

    # Concatenate target data
    df_overall = pd.DataFrame()
    for interval in time_intervals:
        s, e = interval
        df_m_interval = df_m[(s <= df_m.index) & (df_m.index <= e)]

        df_overall = pd.concat([df_overall, df_m_interval])

    df_overall.sort_index(inplace=True)

    if len(type_tag) < 1:
        # Draw all if not specified
        tag_list = np.unique(df_m_interval.type.values)
    else:
        tag_list = type_tag

    for tagname in tag_list:
        if tagname == 'oe':
            continue

        df_target = df_overall[df_overall.type == tagname]
        df_target = handle_postprocessing_for_each_evaluation(
            df_target.copy(), tagname)
        make_cdf_graph(df_target.value.astype(
            float), tagname, output_dir, suffix)


def handle_postprocessing_for_each_evaluation(df, type_tag):
    if type_tag == 'eag':
        def decode_value_eag(row):
            s_list = row.split(';')
            s_list = [float(s) for s in s_list]
            return s_list

        values_series = df['value'].apply(decode_value_eag)
        values_arr = np.stack(values_series.to_numpy())
        t_eag_arr = values_arr[:, 0]/values_arr[:, 1]

        df['value'] = t_eag_arr

        return df

    else:
        return df


def ecdf(data):
    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n

    return x, y


def make_cdf_graph(df, type_tag, output_dir, suffix):
    os.makedirs(output_dir, exist_ok=True)
    x, y = ecdf(df)

    plt.title(F'eCDF : {type_tag}')
    plt.plot(x, y)
    plt.savefig(F'{output_dir}eCDF_{type_tag}{suffix}.png')  # .eps
    plt.close()


def main_cl():
    parser = argparse.ArgumentParser()

    parser.add_argument('-eval_middle_files', '-m', nargs='+',
                        help="List of CSV files to evaluate")
    parser.add_argument('-type_tag', '-t', nargs='*', default=[])

    parser.add_argument('-sections_filename', '-s')
    parser.add_argument('--output_dir', '-o', type=str, default="./")
    parser.add_argument('--suffix', type=str, default="")

    args = parser.parse_args()

    main(args.eval_middle_files, args.type_tag,
         args.output_dir, args.suffix, args.sections_filename)


if __name__ == '__main__':
    main_cl()
