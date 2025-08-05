#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
# import cv2

from evaltools.com_tools.frfw_tools import *
from evaltools.eval_func import bitmap_tools

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

COLOR_LIST = ["black", "red", "blue", "greed", "orange"]


def main(combination_table, est_dir, gt_dir, evaluation_setting=None, output_path='./'):
    # Load combination_table
    comb_set = pd.read_csv(combination_table, header=0,
                           dtype={'data_name': str})
    comb_set.fillna("", inplace=True)
    comb_set = check_combination_dup(comb_set)

    if evaluation_setting is None:
        # Use default if not provided
        evaluation_setting = F'{parent_dir}{os.sep}evaluation_setting.json'
        evaluation_setting = load_json(evaluation_setting)

    bitmap_path = evaluation_setting['OE'][1]['bitmap_path']
    map_origin = evaluation_setting['OE'][1]['O']
    map_ppm = 1/evaluation_setting['OE'][1]['scale']

    for _, row in comb_set.iterrows():
        est = row.est1
        gt = row.gt1
        dataname = row.data_name

        est_filename = F'{est_dir}'
        est_filename += row.est1 if '.csv' in row.est1 else row.est1 + '.csv'
        gt_filename = F'{gt_dir}'
        gt_filename += row.gt1 if '.csv' in row.gt1 else row.gt1 + '.csv'

        # load data
        df_est = load_csv(est_filename)
        df_gt = load_csv(gt_filename)
        bitmap_array = bitmap_tools.load_bitmap_to_ndarray(bitmap_path)

        plot_traj(df_est, df_gt, bitmap_array, dataname, map_origin,
                  map_ppm, output_path)


def check_combination_dup(comb_set):
    """
    Remove duplicate combinations from the combination table

    Parameters
    ----------
    comb_set : pandas.DataFrame
        combination_table, columns: [est1, gt1, tag1, est2, gt2, tag2, section, data_name, rel_target]
    """
    new_comb = comb_set[["est1", "gt1", "data_name"]]
    new_comb_set = new_comb.drop_duplicates(subset=["est1", "gt1"])

    return new_comb_set


def plot_traj(df_est, df_gt, bitmap_array, dataname, map_origin=(-5.625, -12.75), map_ppm=100, output_path='./'):
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    plot_map(ax, bitmap_array, map_origin, map_ppm)

    scatter = ax.scatter(df_gt.x, df_gt.y, c="k", alpha=0.5,
                         s=3, label="location (ground truth)")

    plot_yaw(ax, df_est, decimation_rate=10)
    ax.plot(df_est.x, df_est.y, "gray")
    scatter = ax.scatter(df_est.x, df_est.y, c=df_est.index,
                         s=5, label="location (Estimated)")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    plt.colorbar(scatter, ax=ax, label='timestamp (s)')

    plt.legend()
    plt.savefig(F'{output_path}{dataname}_traj.png', dpi=200)
    plt.close()


def plot_map(ax, bitmap, map_origin, map_ppm):
    bitmap = bitmap.T
    height, width = bitmap.shape[:2]

    # Calculate extent in meters
    width_m = width / map_ppm
    height_m = height / map_ppm

    extent = [
        map_origin[0],                    # left (x_min)
        map_origin[0] + width_m,          # right (x_max)
        map_origin[1] + height_m,                    # bottom (y_min)
        map_origin[1]           # top (y_max)
    ]

    ax.imshow(bitmap, extent=extent, alpha=0.5, cmap='gray')


def plot_yaw(ax, df_gt, decimation_rate=1):
    decimated_df = df_gt.iloc[::decimation_rate]
    arrow_length = 1.5
    dx = arrow_length * np.cos(decimated_df.yaw)
    dy = arrow_length * np.sin(decimated_df.yaw)

    ax.quiver(decimated_df.x, decimated_df.y, dx, dy,
              angles='xy', scale_units='xy', scale=1,
              color='red', alpha=0.7, width=0.003, label="yaw direction (decimated)")


# def _main(pgm_path, map_origin, filename_list, output_filename, title_txt='traj'):
#     pgm_map_array = cv2.imread(pgm_path)
#     fig, ax = plt.subplots(1, 1, figsize=[30, 15], dpi=300)
#     ax.imshow(pgm_map_array)
#     ax.set_title(F'{title_txt}')
#     ax.set_xlim([50*20, 100*20])
#     ax.set_ylim([50*20, 20*20])
#     for idx, filename in enumerate(filename_list):
#         clr, marker = color_select(idx, title_txt)
#         ax = plot_traj(filename, pgm_map_array, ax, clr, map_origin, marker)

#     ax.xaxis.set_major_formatter(FuncFormatter(lambda val, pos: f'{val/20}'))
#     ax.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: f'{val/20}'))

#     ax.set_xlabel('x (m)')
#     ax.set_ylabel('y (m)')

#     fig.tight_layout()
#     fig.legend()
#     fig.savefig(F'{output_filename}', dpi=300)


# def plot_traj(filename, pgm_map_array, ax, clr, map_origin, marker='o'):
#     df = load_csv(filename)
#     fname = filename.split(os.sep)[-1].split('.')[0]

#     X, Y = transform_opencv(df.x.values, df.y.values,
#                             len(pgm_map_array), map_origin)

#     ax.scatter(X, Y, edgecolors=clr, facecolors='none',
#                label=fname, s=3, marker=marker)

#     return ax


def load_csv(filename):
    """
    text(filename) -> pd.read_csv
    other -> return
    """
    if type(filename) == str:
        return pd.read_csv(filename, index_col=0)

    return filename


def transform_opencv(x, y, mapY_len, O_map=(0, 0)):
    return ((x - O_map[0])*20).astype(int), mapY_len - ((y - O_map[1])*20).astype(int)


def color_select(color_idx, title_txt):
    if title_txt == "traj_1":
        if color_idx in [0, 1, 2]:
            clr = 'red'
            if color_idx == 0:
                marker = 'o'
            elif color_idx == 1:
                marker = '^'
            elif color_idx == 2:
                marker = 's'
        elif color_idx in [3, 4, 5]:
            clr = 'blue'
            if color_idx == 3:
                marker = 'o'
            elif color_idx == 4:
                marker = '^'
            elif color_idx == 5:
                marker = 's'

    elif title_txt == "traj_2":
        if color_idx == 0:
            clr = 'red'
            marker = 'o'
        elif color_idx == 1:
            clr = 'blue'
            marker = 'o'

    elif title_txt == "traj_3":
        if color_idx == 0:
            clr = 'red'
            marker = 'o'
        elif color_idx in [1, 2]:
            clr = 'blue'
            if color_idx == 1:
                marker = 'o'
            elif color_idx == 2:
                marker = '^'

    elif title_txt == "traj_4":
        if color_idx in [0, 1, 2]:
            clr = 'red'
            if color_idx == 0:
                marker = 'o'
            elif color_idx == 1:
                marker = '^'
            elif color_idx == 2:
                marker = 's'
        elif color_idx in [3, 4, 5]:
            clr = 'blue'
            if color_idx == 3:
                marker = 'o'
            elif color_idx == 4:
                marker = '^'
            elif color_idx == 5:
                marker = 's'
        elif color_idx in [6, 7, 8, 9]:
            clr = 'green'
            if color_idx == 6:
                marker = 'o'
            elif color_idx == 7:
                marker = '^'
            elif color_idx == 8:
                marker = 's'
            elif color_idx == 9:
                marker = 'D'
        elif color_idx in [10, 11, 12, 13]:
            clr = 'orange'
            if color_idx == 10:
                marker = 'o'
            elif color_idx == 11:
                marker = '^'
            elif color_idx == 12:
                marker = 's'
            elif color_idx == 13:
                marker = 'D'

    else:
        clr = COLOR_LIST[color_idx+1]
        marker = 'o'
    return (clr, marker)


def main_cl():
    parser = argparse.ArgumentParser()

    parser.add_argument('-csv_filename', '-f', nargs='+',
                        help='List of CSV filenames')
    parser.add_argument('--title_txt', default='traj',
                        help='Title of the output plot')
    parser.add_argument('--output_filename', '-o',
                        default='2d_map.png', help='Filename to output results')
    parser.add_argument('-combination_table', '-t')
    parser.add_argument('-est_dir', '-ed')
    parser.add_argument('-gt_dir', '-gd')
    parser.add_argument('--setting', '-s', default=None,
                        help="Evaluation setting filename")

    args = parser.parse_args()
    main(args.combination_table, args.est_dir, args.gt_dir,
         args.setting, args.output_filename)


if __name__ == '__main__':
    main_cl()
