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

    # # Concatenate target data
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
        df_target = df_overall[df_overall.type == tagname]

        plt.plot(df_target.index.to_numpy(),
                 df_target["value"].to_numpy().astype(float))
        plt.savefig(F'{output_dir}/timeline_{type_tag}{suffix}.png')  # .eps
        plt.close()


def main_cl():
    parser = argparse.ArgumentParser()

    parser.add_argument('-eval_middle_files', '-m', nargs='+',
                        help="List of CSV files to evaluate")
    parser.add_argument('-type_tag', '-t', nargs='*', default=[])

    parser.add_argument('-sections_filename', '-s')
    parser.add_argument('--output_dir', '-o', type=str, default=".")
    parser.add_argument('--suffix', type=str, default="")

    args = parser.parse_args()

    main(args.eval_middle_files, args.type_tag,
         args.output_dir, args.suffix, args.sections_filename)


if __name__ == '__main__':
    main_cl()
