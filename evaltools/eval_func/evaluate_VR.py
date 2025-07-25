#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import argparse

import numpy as np


def eval_vr(pfloc, method='Median'):
    """
    Error evaluation of the velocity vector angular change
    """
    V0 = pfloc[['x', 'y']].iloc[0:-2].values
    V1 = pfloc[['x', 'y']].iloc[1:-1].values
    V2 = pfloc[['x', 'y']].iloc[2:].values

    VecA = V1 - V0
    VecB = V2 - V1

    cos = multi_dot(VecA, VecB)/(np.sqrt(multi_dot(VecA, VecA))
                                 * np.sqrt(multi_dot(VecB, VecB)))
    cos = cos[np.logical_not(np.isnan(cos))]
    rad = np.arccos(cos)

    if method == 'Median' or method == 'median':
        return np.median(rad)
    elif method == 'RMSE' or method == 'rmse':
        return rmse(rad)
    else:
        print('method error')
        return None


def multi_dot(A, B):
    return np.sum(A*B, axis=1)


def rmse(ndarray, ndarray0=0):
    return float(np.sqrt(np.mean(np.square(ndarray - ndarray0))))


def main_cl(args):
    eval_vr(args.steps, args.best_param)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-steps', '-s', default='')
    parser.add_argument('-best_param', '-b', default='')

    args = parser.parse_args()
    main_cl(args)
