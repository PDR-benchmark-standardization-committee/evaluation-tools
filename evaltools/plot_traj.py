#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import cv2

COLOR_LIST = ["black", "red", "blue", "greed", "orange"]


def main(pgm_path, map_origin, filename_list, output_filename, title_txt='traj'):
    pgm_map_array = cv2.imread(pgm_path)
    fig, ax = plt.subplots(1, 1, figsize=[30, 15], dpi=300)
    ax.imshow(pgm_map_array)
    ax.set_title(F'{title_txt}')
    ax.set_xlim([50*20, 100*20])
    ax.set_ylim([50*20, 20*20])
    for idx, filename in enumerate(filename_list):
        clr, marker = color_select(idx, title_txt)
        ax = plot_traj(filename, pgm_map_array, ax, clr, map_origin, marker)

    ax.xaxis.set_major_formatter(FuncFormatter(lambda val, pos: f'{val/20}'))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: f'{val/20}'))

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')

    fig.tight_layout()
    fig.legend()
    fig.savefig(F'{output_filename}', dpi=300)


def plot_traj(filename, pgm_map_array, ax, clr, map_origin, marker='o'):
    df = load_csv(filename)
    fname = filename.split(os.sep)[-1].split('.')[0]

    X, Y = transform_opencv(df.x.values, df.y.values,
                            len(pgm_map_array), map_origin)

    ax.scatter(X, Y, edgecolors=clr, facecolors='none',
               label=fname, s=3, marker=marker)

    return ax


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

    parser.add_argument('--pgm_path', '-p', help='Path to the PGM file')
    parser.add_argument('--map_origin', '-m', nargs=2, type=float,
                        default=[-53.973315, -54.523181], help='Origin of the map (O_map). --map-origin {x(m)} {y(m)}')
    parser.add_argument('-csv_filename', '-f', nargs='+',
                        help='List of CSV filenames')
    parser.add_argument('--title_txt', '-t', default='traj',
                        help='Title of the output plot')
    parser.add_argument('--output_filename', '-o',
                        default='2d_map.png', help='Filename to output results')

    args = parser.parse_args()
    main(args.pgm_path, tuple(args.map_origin),
         args.csv_filename, args.output_filename, args.title_txt)


if __name__ == '__main__':
    main_cl()
