#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import argparse

import numpy as np

import evaltools.eval_func as evt  # ref : __init__.py
from evaltools.eval_func import bitmap_tools


class EvaluationHub():

    def __init__(self, eval_list):
        self.eval_list = eval_list
        self.eval_results = {'overall': {}}

    def do_evaluation(self, step, pfloc=None, pfloc_pdr=None):
        """
        Sequentially execute the evaluations listed in self.eval_list.

        By default, the target trajectory for evaluation is step['pfloc'],  
        but if the `pfloc` argument is explicitly specified, it takes precedence.

        *pfoc_pdr : Positioning result without absolute correction, used specifically for EAG evaluation.
        """
        pfloc = step['pfloc'].copy(
            deep=True) if pfloc is None else pfloc.copy(deep=True)
        floors = np.unique(step['gt']['data'].floor) if 'gt' in step.keys(
        ) else np.unique(pfloc.floor.values())
        for floor in floors:
            if floor not in self.eval_results.keys():
                self.eval_results[floor] = {}

            for eval_name, value in self.eval_list.items():
                if value[0]:
                    if eval_name == 'EAG':
                        if pfloc_pdr is None:
                            self.eval_results[floor][eval_name] = None
                            continue
                        pfloc = pfloc_pdr.copy(deep=True)
                    elif eval_name == 'OE':
                        value[1][0] += F'/{floor}_0.01_0.01.bmp'
                        value[1].append(floor)

                    self.eval_traj(eval_name, pfloc, step, value[1], floor)

        return self.eval_results

    def eval_traj(self, eval_name, pfloc, step, params_eval=[], floor='Floor'):
        result = None

        try:
            if eval_name == 'CE':
                result = evt.eval_ce(pfloc, step)
            elif eval_name == 'EAG':
                result = evt.eval_eag(pfloc, step)
            elif eval_name == 'CA':
                result = evt.eval_ca(pfloc, step)
            elif eval_name == 'VE':
                if params_eval[0] == 'negative':
                    result = evt.eval_ve(pfloc)
                elif params_eval[0] == 'positive':
                    result = evt.eval_ve(pfloc, step=step)
            elif eval_name == 'OE':
                result = evt.eval_oe(
                    pfloc, step, params_eval[1], params_eval[0])
            elif eval_name == 'BE':
                result = evt.eval_be(
                    pfloc, step, method=params_eval[0], R=params_eval[1])
            elif eval_name == 'RE':
                result = evt.eval_re(
                    pfloc, step, method=params_eval[0], R=params_eval[1], thresh=params_eval[2], deltaT=params_eval[3])
            elif eval_name == 'PE':
                result = evt.eval_pe(pfloc, step, target_rate=params_eval[0])
            elif eval_name == 'DSE':
                result = evt.eval_dse(
                    pfloc, step, fillna=params_eval[0], mode=params_eval[1], n_jobs=params_eval[2])
            elif eval_name == 'BEM':
                result = evt.eval_bem(pfloc, step, nEstVar=params_eval[0])
            elif eval_name == 'VR':
                result = evt.eval_vr(pfloc, method=params_eval[0])

            # Store results for each floor
            if eval_name not in self.eval_results[floor].keys():
                self.eval_results[floor][eval_name] = result
            else:
                self.eval_results[floor][eval_name] += result
            # Store results without floor distinction
            if eval_name not in self.eval_results['overall'].keys():
                self.eval_results['overall'][eval_name] = result
            else:
                self.eval_results['overall'][eval_name] += result

        except KeyError as ke:
            print(F'【Eval Error in {eval_name}】: {ke}')

    def eval_switcher(self, eval_name, pfloc, step, params_eval=[]):
        result = None

        try:
            if eval_name == 'CE':
                result = evt.eval_ce(pfloc, step, quantile=params_eval[0])
            elif eval_name == 'EAG':
                result = evt.eval_eag(pfloc, step)
            elif eval_name == 'CA':
                result = evt.eval_ca(pfloc, step)
            elif eval_name == 'VE':
                if params_eval[0] == 'negative':
                    result = evt.eval_ve(pfloc)
                elif params_eval[0] == 'positive':
                    result = evt.eval_ve(pfloc, step=step)
            elif eval_name == 'OE':
                result = evt.eval_oe(
                    pfloc, step, params_eval[1], params_eval[0])
            elif eval_name == 'BE':
                result = evt.eval_be(
                    pfloc, step, method=params_eval[0], R=params_eval[1])
            elif eval_name == 'RE':
                result = evt.eval_re(
                    pfloc, step, method=params_eval[0], R=params_eval[1], thresh=params_eval[2], deltaT=params_eval[3])
            elif eval_name == 'PE':
                result = evt.eval_pe(pfloc, step, target_rate=params_eval[0])
            elif eval_name == 'DSE':
                result = evt.eval_dse(
                    pfloc, step, fillna=params_eval[0], mode=params_eval[1], n_jobs=params_eval[2])
            elif eval_name == 'BEM':
                result = evt.eval_bem(pfloc, step, nEstVar=params_eval[0])
            elif eval_name == 'VR':
                result = evt.eval_vr(pfloc, method=params_eval[0])

        except KeyError as ke:
            print(F'【Eval Error in {eval_name}】: {ke}')

        return result

    def eval_switcher_simple(self, eval_name, est, gt, params_eval=[]):
        """
        Execute the specified absolute localization evaluation function

        Parameters
        ----------
        eval_name : String
            Evaluation function name
        est : pandas.DataFrame
            Estimated trajectory1, columns, [timestamp, x, y, (z,)]
        gt : pandas.DataFrame
            Ground-truth trajectory corresponding to `est1`, columns, [timestamp, x, y, (z,)]
        params_eval : List
            Parameters provided to each evaluation function
        """
        result = None

        try:
            if eval_name == 'CE':
                result = evt.eval_CE_tl(gt, est)
            elif eval_name == 'SE':
                result = evt.eval_SE_tl(gt, est)
            elif eval_name == 'EAG':
                result = evt.eval_EAG_tl(gt, est, **params_eval)
            elif eval_name == 'CA':
                result = evt.eval_CA_tl(gt, est, **params_eval)
            elif eval_name == 'VE':
                if params_eval['is_neg']:
                    gt = None
                result = evt.eval_VE_tl(est, df_gt=gt)
            elif eval_name == 'OE':
                if "floor" in est.columns:
                    floor = np.unique(est.floor)[0]
                    bitmap = bitmap_tools.load_bitmap_to_ndarray(
                        params_eval['bitmap_path'][floor])
                else:
                    bitmap = bitmap_tools.load_bitmap_to_ndarray(
                        params_eval['bitmap_path'])

                bitmap_scale = params_eval['scale']

                result, _ = evt.eval_OE_tl(
                    est, bitmap, bitmap_scale, params_eval["O"])
            elif eval_name == 'FE':
                result = evt.eval_FE_tl(gt, est)

        except KeyError as ke:
            print(F'【Eval Error in {eval_name}】: {ke}')
        except FileNotFoundError as ffe:
            print(F'【Eval Error in {eval_name}】: {ffe}')
        except Exception as e:
            print(F'【Eval Error in {eval_name}】: {e}')

        return result

    def eval_switcher_simple_rel(self, eval_name, id1, id2, est1, est2, gt1, gt2, params_eval=[], setname=''):
        """
        Execute the specified relative evaluation function

        Parameters
        ----------
        eval_name : String
            Evaluation function name
        id1 : String
            ID to distinguish between vector directions (AB vs. BA) 
        id2 : String
            ID to distinguish between vector directions (AB vs. BA)
        est1 : pandas.DataFrame
            Estimated trajectory1, columns, [timestamp, x, y, (z,) yaw] or [timestamp, x, y, (z,) qx, qy, qz, qw]
        est2 : pandas.DataFrame
            Estimated trajectory2, columns, [timestamp, x, y, (z,) yaw] or [timestamp, x, y, (z,) qx, qy, qz, qw]
        gt1 : pandas.DataFrame
            Ground-truth trajectory corresponding to `est1`, columns, [timestamp, x, y, (z,) yaw] or [timestamp, x, y, (z,) qx, qy, qz, qw]
        gt2 : pandas.DataFrame
            Ground-truth trajectory corresponding to `est2`, columns, [timestamp, x, y, (z,) yaw] or [timestamp, x, y, (z,) qx, qy, qz, qw]

        Returns
        -------
        result: pandas.DataFrame
            Error at each timestamp, columns: [timestamp, type, value]
        """
        result = None

        try:
            if eval_name == 'RDA':
                result = evt.evaluate_RDA_tl(est1, est2, gt1, gt2, setname)
            elif eval_name == 'RHA':
                result = evt.evaluate_RHA_tl(est1, est2, gt1, gt2, setname)
            elif eval_name == 'RPA':
                result = evt.evaluate_RPA_tl(est1, est2, gt1, gt2, id1, id2)

        except Exception as e:
            print(e)

        return result


def main(pfloc, step, params_eval):
    """
    """

    pass


def main_cl(args):
    main()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    main_cl(args)
