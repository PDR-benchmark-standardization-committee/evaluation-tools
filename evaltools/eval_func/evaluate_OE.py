#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
import numpy as np
import shapely
# from tqdm import tqdm
# tqdm.pandas()
import matplotlib.pyplot as plt

from evaltools.eval_func import bitmap_tools

DEBUG = False


def eval_OE_tl_multifloor(df_est, obstacle_dict, bitmap_scale=0.01, O=(0, 0)):
    df_oe_tl = None
    for floor, obstacle in obstacle_dict.items():
        df_oe_tl_floor = eval_OE_tl(df_est, obstacle, bitmap_scale, O)
        df_oe_tl_floor['type'] = F'oe_{floor}'

        if df_oe_tl is None:
            df_oe_tl = df_oe_tl_floor
        else:
            df_oe_tl = pd.concat([df_oe_tl, df_oe_tl_floor])

    return df_oe_tl


def eval_OE_tl(df_est, obstacle, bitmap_scale=0.01, O=(0, 0), debug_output='./'):
    """
    Calculate Obstacle Error

    Parameters
    ----------
    df_est : pandas.DataFrame
        Estimated position, columns: [timestamp, x, y, floor, ...]
    obstacle : bitmap
        0 : movable, 1 : obstacle
    bitmap_scale : float
        scale of bitmap ex) 1/100(m), 1/25(m), 1/10(m)

    Returns
    ----------
    result : pandas.DataFrame
        Error at each timestamp, columns: [timestamp, type, value]
        value: True (in obstacle) or False (out obstacle)
    """
    obstacle = np.logical_not(np.transpose(obstacle))
    # print(len(obstacle[0]), len(obstacle))
    # x_block_m = len(obstacle[0]) / map_size[0]
    # y_block_m = len(obstacle) / map_size[1]
    block_m = 1/bitmap_scale

    # print(block_m)

    # Convert bitmap coordinate to mapsize coordinate
    df_est.dropna(subset=['x', 'y'], inplace=True)

    df_est['x_block_num'] = (
        (df_est['x'] - O[0]) * block_m).map(lambda x: int(x) - 1 if int(x) != 0 else 0)
    df_est['y_block_num'] = (
        (df_est['y'] - O[1]) * block_m).map(lambda x: int(x) - 1 if int(x) != 0 else 0)

    # print(df_est.iloc[0])
    df_est['unixtime'] = df_est.index
    df_est = df_est[['unixtime', 'x', 'y', 'x_block_num', 'y_block_num']]

    x_block_num_dif = df_est['x_block_num'].diff()
    y_block_num_dif = df_est['y_block_num'].diff()
    x_block_num_dif[-1] = 0
    y_block_num_dif[-1] = 0
    x_block_num_dif.dropna(inplace=True)
    y_block_num_dif.dropna(inplace=True)

    df_est['x_block_num_dif'] = (x_block_num_dif.astype(int).values)
    df_est['y_block_num_dif'] = (y_block_num_dif.astype(int).values)

    # Auxiliary function to calculate E_obstacle
    # check_pattern, is_inside_map, is_obstacle, is_obstacle_around, is_obstacle_exist,
    # ObstacleCordinate_count, CheckCordinate_count

    def check_pattern(row):
        '''
        Fucntion to calculate obstacle error
        Appoint pattern to trajection point
        Parameters
        ----------
        row : pd.Series
            trajection file row data
        Return
        ------
        pattern : str
            'A' or 'B' or 'C' or 'D' or 'E'
        '''

        x_dif = row['x_block_num_dif']
        y_dif = row['y_block_num_dif']

        if x_dif == 0:
            if y_dif > 0:
                return 'A'
            elif y_dif == 0:
                return 'E'
            else:
                return 'B'

        else:
            if x_dif > 0:
                return 'C'
            elif x_dif < 0:
                return 'D'

    def is_inside_map(x, y):
        '''
        Fucntion to calculate obstacle error
        Check wheather input cordinate is inside bitmap data or not
        Parameters
        ----------
        x, y : int
            Cordinates

        Returns
        -------
        boolean : bool
            if cordinate is inside bitmap : True, else :  False
        '''
        if 0 <= x < obstacle.shape[1] and 0 <= y < obstacle.shape[0]:
            return True
        else:
            return False

    def is_obstacle(x, y):
        '''
        Fucntion to calculate obstacle error
        Check wheather obstacle exsits on input cordinates in bitmap data or not

        Parameters
        ----------
        x, y : int
            Cordinates

        Returns
        -------
        boolean : bool
            if obstacle exists on input cordinates : True, else :  False
        '''
        if obstacle[y][x] == 1:
            return True
        else:
            return False

    def is_obstacle_around(x, y):
        '''
        Fucntion to calculate obstacle error
        Check wheather all area around input cordinates are filled with obstacle or not

        Parameters
        ----------
        x, y : int
            Cordinates

        Returns
        -------
        boolean : bool
            if no empty point exist : True, else :  False
        '''

        for x_i in range(-3, 4):
            for y_i in range(-3, 4):
                if is_inside_map(x + x_i, y + y_i):
                    if not is_obstacle(x + x_i, y + y_i):
                        return False
        return True

    def is_obstacle_exist(x, y):
        '''
        Fucntion to calculate obstacle error
        Check wheather obstacle exist on input cordinate including around area
        Parameters
        ----------
        x, y : int
            Cordinates

        Returns
        -------
        boolean : bool
            if obstacle exist on input cordinates: True, else :  False
        '''
        if is_inside_map(x, y):
            if is_obstacle(x, y):
                if is_obstacle_around(x, y):
                    # print("is obstacle around")
                    return True
            return False
        else:
            return True

    # def is_obstacle_exist(x, y):
    #     '''
    #     Fucntion to calculate obstacle error
    #     Check wheather obstacle exist on input cordinate including around area
    #     Parameters
    #     ----------
    #     x, y : int
    #         Cordinates

    #     Returns
    #     -------
    #     boolean : bool
    #         if obstacle exist on input cordinates: True, else :  False
    #     '''
    #     if is_inside_map(x, y):
    #         if is_obstacle(x, y):
    #             if is_obstacle_around(x, y):
    #                 return True

    #     return True

    def ObstacleCoordinate_count(row):
        '''
        Fucntion to calculate obstacle error
        Count total coordinates where obstacle exist in trajection data
        Parameters
        ----------
        row : pd.Series
            trajection file row data

        Returns
        -------
        obstacle_count : int
            number of total coordinates where obstacle exist
        '''

        y_block_num = row['y_block_num']
        y_block_num_t1 = y_block_num + row['y_block_num_dif']
        # y_block_num_t1 = y_block_num - row['y_block_num_dif']

        x_block_num = row['x_block_num']
        x_block_num_t1 = x_block_num + row['x_block_num_dif']
        # x_block_num_t1 = x_block_num - row['x_block_num_dif']

        obstacle_count = 0

        if row['pattern'] == 'A':
            for y in range(y_block_num, y_block_num_t1):
                if is_obstacle_exist(x_block_num, y):
                    obstacle_count += 1

        elif row['pattern'] == 'B':
            for y in range(y_block_num, y_block_num_t1, -1):
                if is_obstacle_exist(x_block_num, y):
                    obstacle_count += 1

        elif row['pattern'] == 'C':
            a = int((y_block_num - y_block_num_t1) /
                    (x_block_num - x_block_num_t1))
            b = y_block_num - (a * x_block_num)
            for x in range(x_block_num, x_block_num_t1):
                y = int(a * x + b)
                if is_obstacle_exist(x, y):
                    obstacle_count += 1

        elif row['pattern'] == 'D':
            a = int((y_block_num - y_block_num_t1) /
                    (x_block_num - x_block_num_t1))
            b = y_block_num - (a * x_block_num)
            for x in range(x_block_num, x_block_num_t1, -1):
                y = int(a * x + b)
                if is_obstacle_exist(x, y):
                    obstacle_count += 1

        elif row['pattern'] == 'E':
            x = x_block_num
            y = y_block_num
            if is_obstacle_exist(x, y):
                obstacle_count += 1

        return obstacle_count

    df_est['pattern'] = df_est.apply(check_pattern, axis=1)

    # obstacle_coordinate_count =  df_est.progress_apply(ObstacleCoordinate_count, axis=1)

    obstacle_coordinate_count = df_est.apply(ObstacleCoordinate_count, axis=1)

    TorF_list = [True if obs_cout > 0 else False
                 for obs_cout in obstacle_coordinate_count]

    if DEBUG:
        print('debug')
        df_est['obstacle_coordinate_count'] = obstacle_coordinate_count
        df_est.to_csv('debug_OE.csv')
        check_error(df_est, obstacle, obstacle_coordinate_count, debug_output)

    df_oe_tl = pd.DataFrame(data={'timestamp': df_est.index,
                            'type': 'oe', 'value': TorF_list}).set_index('timestamp')
    return (df_oe_tl, obstacle_coordinate_count)


def eval_OE(df_est, step, floor, bitmap_path):
    bitmap, mapsize = gen_runtime_OE(step, floor, bitmap_path)

    df_oe_tl, obstacle_coordinate_count = eval_OE_tl(
        df_est, bitmap, mapsize, draw_flg=True)

    def CheckCordinate_count(row):
        '''
        Fucntion to calculate obstacle error
        Count total codinates checked wheather obstacle exist or not
        Parameters
        ----------
        row : pd.Series
            trajection file row data

        Returns
        -------
        check_coordinate_count : int
            number of total cordinates checked wheather obstacle exist or not
        '''

        pattern = row['pattern']
        if pattern == 'A' or pattern == 'B':
            return abs(row['y_block_num_dif'])
        else:
            return abs(row['x_block_num_dif'])

    check_coordinate_count = df_est.apply(CheckCordinate_count, axis=1)

    obstacle_check = pd.DataFrame({'check_coordinate_count': list(check_coordinate_count),
                                  'obstacle_coordinate_count': list(obstacle_coordinate_count)})

    # return np.sum(list(obstacle_coordinate_count))/np.sum(list(check_coordinate_count))
    OE = (np.sum(list(check_coordinate_count)) - np.sum(list(obstacle_coordinate_count))
          )/np.sum(list(check_coordinate_count))  # acc
    return OE


def _main_multi_floor(df_est, obstacle_set_dict, draw_flg=False, output_path='./output/'):
    SUM = 0
    ERR = 0
    for floor, v in obstacle_set_dict.items():
        df_est_floor = df_est[df_est.floor == floor]
        OE, all, error = eval_OE(
            df_est_floor, v[0], v[1], v[2], draw_flg, output_path=output_path+F'{floor}/')

        SUM += all
        ERR += error

    return ERR/SUM


def calc_mapsize(geom):
    x_min, y_min, x_max, y_max = shapely.bounds(geom)
    return ((x_max - x_min) * 1, (y_max - y_min) * 1)


def check_error(df_est, obstacle, obstacle_cordinate_count, output_path='./output/'):
    mask_1 = (0 < obstacle_cordinate_count)
    mask_0 = (obstacle_cordinate_count == 0)

    plt.rcParams['image.cmap'] = 'viridis'

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # ax.pcolor(np.flipud(obstacle))
    ax.pcolor(obstacle)
    ax.scatter(df_est['x_block_num'][mask_0].values, df_est['y_block_num']
               [mask_0].values, s=1, color='black', label='movable')
    ax.scatter(df_est['x_block_num'][mask_1].values, df_est['y_block_num']
               [mask_1].values, s=1, color='red', label='in obstacle')

    # Target position index (for iloc)
    target_pos = df_est.index.get_indexer_for(df_est[mask_1].index)

    for i in target_pos:
        if i + 1 < len(df_est):
            x1, y1 = df_est.iloc[i][['x_block_num', 'y_block_num']]
            x2, y2 = df_est.iloc[i + 1][['x_block_num', 'y_block_num']]
            ax.plot([x1, x2], [y1, y2], color='red', linewidth=0.5, linestyle='--',
                    label='interpolation' if i == target_pos[0] else "")

    x_ticks = ax.get_xticks()
    y_ticks = ax.get_yticks()
    ax.set_xticklabels(x_ticks/100)
    ax.set_yticklabels(y_ticks/100)  # Convert scale from cm to m
    ax.set_xlabel('(m)')
    ax.set_ylabel('(m)')
    ax.legend()

    os.makedirs(output_path, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path + 'OE.png')
    plt.close()


def gen_runtime_OE(step, floor, bitmap_path=None):
    """
    Generate data required to run the obstacle evaluation function
    """
    if 'occupancy_grids' in step['gis'].keys():
        bitmap = step['gis']['occupancy_grids']['grids_dict'][floor]
        dx = step['gis']['occupancy_grids']['dx']
        dy = step['gis']['occupancy_grids']['dy']
        mapsize = (len(bitmap[0]) * dx, len(bitmap) * dy)

    elif 'floor' in step['gis'].keys() and 'geom' in step['gis']['floor'].keys():
        try:
            bitmap = bitmap_tools.load_bitmap_to_ndarray(bitmap_path)
            geom = step['gis']['floor']['geom']
            mapsize = (geom.bounds[0], geom.bounds[1])
        except FileNotFoundError:
            raise FileNotFoundError('[bitmap] is not found')

    else:
        raise KeyError(
            'Both [gis.occupancy_grids] and [gis.floor.geom] are not found')

    return (bitmap, mapsize)
