#!/usr/bin/env python
# -*- coding: utf-8 -*-
from evaltools.evaluation_hub import EvaluationHub
from evaltools.com_tools.frfw_tools import *
import pandas as pd
import numpy as np
import os
import sys
import argparse
import gc
import logging
from logging import getLogger
logger = getLogger(__name__)


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))


def main_with_dir(est_dir, gt_dir, combination_table, output_result="evaluation_result.csv", evaluation_setting=None):
    """
    xDR Challenge 2025

    Parameters
    ----------
    combination_table : String
        Filename of the table listing est-gt combinations
    """
    # Explicitly delete old file because mode='a'
    if os.path.exists(F'{output_result}'):
        os.remove(F'{output_result}')

    if evaluation_setting is None:
        # Use default if not provided
        evaluation_setting = F'{parent_dir}{os.sep}evaluation_setting.json'

    # Load combination_table
    comb_set = pd.read_csv(combination_table, header=0,
                           dtype={'data_name': str})
    comb_set.fillna("", inplace=True)
    comb_set = check_combination_dup(comb_set)

    result_overall = None
    header = True
    counter_oom = 1
    error_rows = []
    error_log_path = 'error_log_do_eval.csv'
    for _, row in comb_set.iterrows():
        est = row.est1
        gt = row.gt1
        data_name = row.data_name

        est_filename1 = F'{est_dir}'
        est_filename1 += row.est1 if '.csv' in row.est1 else row.est1 + '.csv'
        gt_filename1 = F'{gt_dir}'
        gt_filename1 += row.gt1 if '.csv' in row.gt1 else row.gt1 + '.csv'

        try:
            print(F'----- {data_name} -----')
            result = main(est_filename1, gt_filename1, evaluation_setting)
            result = add_colums(result, data_name)
        except Exception as e:
            print(f'Error on {data_name}')
            print(e)
            error_info = {
                'data_name': data_name,
                'est_filename1': est_filename1,
                'gt_filename1': gt_filename1,
                'error_msg': str(e)
            }
            error_rows.append(error_info)
            continue

        if result_overall is None:
            result_overall = result
        else:
            result_overall = pd.concat([result_overall, result])

        # OOM prevention
        if counter_oom % 5 == 0:
            result_overall.to_csv(F'{output_result}', mode='a', header=header)
            del result_overall
            gc.collect()
            result_overall = None
            if header:
                header = False

        counter_oom += 1

    result_overall.to_csv(F'{output_result}', mode='a', header=header)
    if error_rows:
        df_error = pd.DataFrame(error_rows)
        df_error.to_csv(error_log_path, index=False)

    return result_overall


def add_colums(result, data_name):
    """
    Add extra columns for xDR_Challenge_2025 to the intermediate file

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
    data_name_list = np.full(len(result), data_name)

    result['data_name'] = data_name_list

    return result


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


def main(est_filename, gt_filename, evaluation_setting=None):
    """
    Perform accuracy evaluation.

    This function compares ground truth and estimated data to compute accuracy metrics.

    Parameters
    ----------
    est_filename : String
        Estimated trajectory filename (.csv), columns: [timestamp, x, y, yaw, floor]
    gt_filename : String
        Ground-Truth trajectory filename (.csv), columns: [timestamp, x, y, yaw, floor]
    evaluation_setting : String
        Evaluation setting filename (.json), format: {eval_name:[bool, param_list], ...}

    Returns
    ----------
    df_evaluation_results_tl : pandas.DataFrame
        Middle data of evaluation results, columns: [timestamp, type, value]
        type: Type of evaluation, {CE, CA, EAG, VE, OE, ...}
        value: Error at each timestamp
    """
    if evaluation_setting is None:
        # Use default if not provided
        evaluation_setting = F'{parent_dir}{os.sep}evaluation_setting.json'

    # load data
    df_est = load_csv(est_filename)
    df_gt = load_csv(gt_filename)

    evaluation_dict = load_json(evaluation_setting)

    # Do evaluation per timestamp
    eh = EvaluationHub(evaluation_dict)
    df_tl = None
    for eval_name, val in eh.eval_list.items():
        logging.debug(F'【{eval_name}】')
        do_eval = val[0]
        param = val[-1]
        if do_eval:
            df_result = eh.eval_switcher_simple(
                eval_name, df_est, df_gt, param)

        if df_tl is None:
            df_tl = df_result
        else:
            df_tl = pd.concat([df_tl, df_result])

    return df_tl


def pickle_rapper(pickle_filename, evaluation_setting):
    """
    main with pickle base
    """
    step = pd.read_pickle(pickle_filename)

    return main(step['pfloc'], step['gt']['data'], evaluation_setting)


def main_cl():
    parser = argparse.ArgumentParser(
        description="Calculates the Absolute Localization Error between estimated and ground-truth trajectories.")

    parser.add_argument('--est', '-e', nargs="*",
                        help="Estimated trajectory filename")
    parser.add_argument('--gt', '-g', nargs="*",
                        help="Ground-Truth trajectory filename")

    parser.add_argument('-est_dir', '-ed', default=None,
                        help="Estimated trajectory dirname")
    parser.add_argument('-gt_dir', '-gd', default=None,
                        help="Ground-Truth trajectory dirname")
    parser.add_argument('--combination_table', '-t', default=None,
                        help="Filename of the table listing est-gt combinations")

    parser.add_argument('--setting', '-s', default=None,
                        help="Evaluation setting filename")
    parser.add_argument('--pickle', '-p', default=None, help="Pickle format")
    parser.add_argument('--output_result', '-o',
                        default="./evaluation_result.csv", help="Output directory name")

    args = parser.parse_args()

    if args.combination_table is not None:
        df_tl = main_with_dir(args.est_dir, args.gt_dir, args.combination_table,
                              args.output_result, args.setting)

    elif args.pickle is None:
        # csv base
        df_tl = main(args.est, args.gt, args.setting)

        # output
        df_tl.to_csv(args.output_result)
    else:
        # pickle base
        df_tl = pickle_rapper(args.pickle, args.setting)

        # output
        df_tl.to_csv(args.output_result)

    print('do_evaluation_tl END')


if __name__ == '__main__':
    main_cl()
