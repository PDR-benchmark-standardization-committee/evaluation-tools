#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import glob
import argparse
import json
import itertools
import gc

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from evaltools.com_tools.frfw_tools import *
from evaltools.com_tools.convert_tools import *
from evaltools.evaluation_hub import EvaluationHub

from evaltools.com_tools import viewer_tools

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

EST_TAG = '_est'
GT_TAG = ''
DEBUG = False


def main_with_dp3(est_dir, gt_dir, evaluation_setting=None):
    """
    Perform relative evaluation for all combinations of trajectories in the specified directory.

    Each pair of Estimated and Ground-truth trajectories must be linked by a common identifier in their filenames.
    Currently, it assumes the naming convention used in xDR_Challenge_2024 (e.g., {dataset}_{id}.csv).
    """
    if evaluation_setting is None:
        # Use default if not provided
        evaluation_setting = F'{parent_dir}{os.sep}evaluation_setting_rel.json'
    est_filenames = glob.glob(est_dir + '*.csv')
    comb_set = itertools.combinations(est_filenames, 2)

    for est_filename1, est_filename2 in comb_set:
        dataname1, id1 = get_filename_info(est_filename1, EST_TAG)
        dataname2, id2 = get_filename_info(est_filename2, EST_TAG)

        gt_filename1 = gt_dir + F'{dataname1}{GT_TAG}.csv'
        gt_filename2 = gt_dir + F'{dataname2}{GT_TAG}.csv'

        result = main(est_filename1, est_filename2,
                      gt_filename1, gt_filename2, evaluation_setting, id1, id2)


def main_with_comb_table(est1_dir, est2_dir, gt1_dir, gt2_dir, combination_table,
                         output_result="evaluation_result_rel.csv", evaluation_setting=None):
    """
    xDR_Challenge_2025

    Perform relative evaluation of trajectory pairs specified in the combination_table
    for the trajectories stored in the specified directory.

    Parameters
    ----------
    est1_dir : String
        Estimated trajectory dirname1
    est2_dir : String
        Estimated trajectory dirname2
    gt1_dir : String
        Ground-truth trajectory dirname1
    gt2_dir : String
        Ground-truth trajectory dirname2
    combination_table : String
        Filename of the table listing est-gt combinations,
        columns: [est1, gt1, tag1, est2, gt2, tag2, section, data_name, rel_target]
    output_result : String
        Output directory name
    evaluation_setting : String
        Relative evaluation setting filename

    Returns
    -------
    result_overall : pandas.DataFrame
        Intermediate file of relative evaluation results (per timestamp),
        collumns : [timestamp, type, value, section, data_name, rel_target]
    """
    # Explicitly delete old file because mode='a'
    if os.path.exists(F'{output_result}'):
        os.remove(F'{output_result}')

    if evaluation_setting is None:
        # Use default if not provided
        evaluation_setting = F'{parent_dir}/evaluation_setting_rel.json'
        print(F'load {evaluation_setting}')

    # Load combination_table
    comb_set = pd.read_csv(combination_table, header=0,
                           dtype={'data_name': str})
    comb_set.fillna("", inplace=True)

    result_overall = None
    header = True
    error_rows = []
    error_log_path = 'error_log.csv'
    for _, row in comb_set.iterrows():
        id1 = row.tag1
        id2 = row.tag2
        section = row.section
        data_name = row.data_name
        rel_target = row.rel_target

        est_filename1 = F'{est1_dir}'
        est_filename1 += row.est1 if '.csv' in row.est1 else row.est1 + '.csv'
        est_filename2 = F'{est2_dir}{rel_target}/{data_name}/'
        est_filename2 += row.est2 if '.csv' in row.est2 else row.est2 + '.csv'
        gt_filename1 = F'{gt1_dir}'
        gt_filename1 += row.gt1 if '.csv' in row.gt1 else row.gt1 + '.csv'
        gt_filename2 = F'{gt2_dir}{rel_target}/{data_name}/'
        gt_filename2 += row.gt2 if '.csv' in row.gt2 else row.gt2 + '.csv'

        try:
            print(F'----- {data_name} {section} {rel_target} -----')
            result = main(est_filename1, est_filename2,
                          gt_filename1, gt_filename2, evaluation_setting, id1, id2)
            result = add_colums(result, section, data_name, rel_target)
        except Exception as e:
            print(
                f'Error on {data_name}, section {section}, target {rel_target}')
            print(e)
            error_info = {
                'data_name': data_name,
                'section': section,
                'rel_target': rel_target,
                'est_filename1': est_filename1,
                'est_filename2': est_filename2,
                'gt_filename1': gt_filename1,
                'gt_filename2': gt_filename2,
                'error_msg': str(e)
            }
            error_rows.append(error_info)
            continue

        if result_overall is None:
            result_overall = result
        else:
            result_overall = pd.concat([result_overall, result])

        # OOM prevention
        if section == "section1":
            result_overall.to_csv(F'{output_result}', mode='a', header=header)
            del result_overall
            gc.collect()
            result_overall = None
            if header:
                header = False

    if result_overall is None:
        return result_overall
    result_overall.to_csv(F'{output_result}', mode='a', header=header)
    if error_rows:
        df_error = pd.DataFrame(error_rows)
        df_error.to_csv(error_log_path, index=False)

    return result_overall


def add_colums(result, section, data_name, rel_target):
    """
    Add extra columns for xDR_Challenge_2025 to the intermediate file

    section     : Target section (for internal validation)
    data_name   : Dataset name
    rel_target  : Destination type (exhibit, robot)

    Parameters
    ----------
    result : pandas.DataFrame
        Middle data of evaluation results, columns: [timestamp, type, value]
    data_name : String
        dataset name 

    Returns
    -------
    result : pandas.DataFrame
        Middle data of evaluation results, columns: [timestamp, type, value]
    """
    section_list = np.full(len(result), section)
    data_name_list = np.full(len(result), data_name)
    rel_target_list = np.full(len(result), rel_target)

    result['section'] = section_list
    result['data_name'] = data_name_list
    result['rel_target'] = rel_target_list

    return result


def main(est_filename1, est_filename2, gt_filename1, gt_filename2, evaluation_setting, id1=1, id2=2):
    """
    Parameters
    ----------
    est_filename1 : String
        Estimated trajectory filename1 (.csv), columns: [timestamp, x, y, yaw, floor]
    est_filename2 : String
        Estimated trajectory filename2 (.csv), columns: [timestamp, x, y, yaw, floor]
    gt_filename1 : String
        Ground-Truth trajectory filename1 (.csv), columns: [timestamp, x, y, yaw, floor]
    gt_filename2 : String
        Ground-Truth trajectory filename2 (.csv), columns: [timestamp, x, y, yaw, floor]
    evaluation_setting : String
        Evaluation setting filename (.json), format: {eval_name:[bool, param_list], ...}
    id1 : String
        ID to distinguish between vector directions (AB vs. BA)
    id2 : String
        ID to distinguish between vector directions (AB vs. BA)

    Retruns
    ----------
    df_evaluation_results_tl : pandas.DataFrame
        Middle data of evaluation results, columns: [timestamp, type, value, section, data_name, rel_target]
        type: Type of evaluation, {RHA, RDA, RPA}
        value: Error at each timestamp
    """
    # Load data
    df_est1 = load_csv(est_filename1)
    df_est2 = load_csv(est_filename2)
    df_gt1 = load_csv(gt_filename1)
    df_gt2 = load_csv(gt_filename2)
    if evaluation_setting is None:
        # Use default if not provided
        evaluation_setting = F'{parent_dir}{os.sep}evaluation_setting_rel.json'
    evaluation_dict = load_json(evaluation_setting)

    df_est1 = check_format(df_est1)
    df_est2 = check_format(df_est2)
    df_gt1 = check_format(df_gt1)
    df_gt2 = check_format(df_gt2)

    if DEBUG:
        plot_traj_with_yaw(df_est1, df_gt1, F'datas/debug/{id1}_{id2}')

    # Do evaluation per timestamp
    eh = EvaluationHub(evaluation_dict)
    df_tl = None
    for eval_name, val in eh.eval_list.items():
        print(F'【{eval_name}】')
        do_eval, param = val
        if do_eval:
            df_result = eh.eval_switcher_simple_rel(
                eval_name, id1, id2, df_est1, df_est2, df_gt1, df_gt2, param)

        if df_tl is None:
            df_tl = df_result
        else:
            df_tl = pd.concat([df_tl, df_result])

    return df_tl


def check_format(df):
    if len(df) < 1:
        raise Exception('df is Empty')

    # if 'timestamp' not in df.columns:
    #     if df.index.name == 'timestamp':
    #         df.index.name = 'ts'
    #         df['timestamp'] = df.index

    if 'yaw' not in df.columns:
        if np.all([clm in df.columns for clm in ['qx', 'qy', 'qz', 'qw']]):
            # print('convert Quaternion to Yaw')
            df = convert_quat_to_yaw(df)

    return df


def pickle_rapper(pickle_filename, evaluation_setting):
    """
    main with pickle base
    """
    step = pd.read_pickle(pickle_filename)

    return main(step['pfloc'], step['gt']['data'], evaluation_setting)


def plot_traj_with_yaw(df1, df2, img_name):
    fig, ax = plt.subplots(figsize=(10, 8))

    ax = viewer_tools.plot_traj_with_yaw(
        ax, df1, 'est', 'royalblue', 'skyblue')
    ax = viewer_tools.plot_traj_with_yaw(
        ax, df2, 'gt', 'darkorange', 'gold', skip_set_options=True)

    fig.savefig(F'{img_name}.png', dpi=200)
    plt.close()


def main_cl():
    parser = argparse.ArgumentParser(
        description="Calculates the Relative Error between estimated and ground-truth trajectories.")

    parser.add_argument('-est1', '-e1', help="Estimated trajectory filename1")
    parser.add_argument('-est2', '-e2', help="Estimated trajectory filename2")
    parser.add_argument(
        '-gt1', '-g1', help="Ground-Truth trajectory filename1")
    parser.add_argument(
        '-gt2', '-g2', help="Ground-Truth trajectory filename2")

    parser.add_argument('-est_dir', '-ed', default=None,
                        help="Estimated trajectory dirname")
    parser.add_argument('-gt_dir', '-gd', default=None,
                        help="Ground-Truth trajectory dirname")

    parser.add_argument('-est1_dir', '-ed1', default=None,
                        help="Estimated trajectory dirname1")
    parser.add_argument('-est2_dir', '-ed2', default=None,
                        help="Estimated trajectory dirname2")
    parser.add_argument('-gt1_dir', '-gd1', default=None,
                        help="Ground-Truth trajectory dirname1")
    parser.add_argument('-gt2_dir', '-gd2', default=None,
                        help="Ground-Truth trajectory dirname2")

    parser.add_argument('--combination_table', '-t', default=None,
                        help="Filename of the table listing est-gt combinations")
    parser.add_argument('--output_result', '-o',
                        default='./evaluation_result_rel.csv', help="Output directory name")

    parser.add_argument('--setting', '-s', default=None,
                        help="Relative evaluation setting filename")
    # parser.add_argument('--pickle', '-p', default=None)

    args = parser.parse_args()

    if args.combination_table is not None:
        df_tl = main_with_comb_table(args.est1_dir, args.est2_dir,
                                     args.gt1_dir, args.gt2_dir,
                                     args.combination_table, args.output_result, args.setting)
    elif args.est_dir is not None and args.gt_dir is not None:
        df_tl = main_with_dp3(args.est_dir, args.gt_dir, args.setting)

        df_tl.to_csv('evaluation_rel_result.csv')
    else:
        _, id1 = get_filename_info(args.est1, EST_TAG)
        _, id2 = get_filename_info(args.est2, EST_TAG)
        df_tl = main(args.est1, args.est2,
                     args.gt1, args.gt2, args.setting, id1, id2)

        df_tl.to_csv('evaluation_rel_result.csv')

    print('do_evaluation_tl_rel END')


if __name__ == '__main__':
    main_cl()
