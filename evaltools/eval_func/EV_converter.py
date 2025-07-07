#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys
import glob
import json
import argparse

### convert Evaluation Values into Normalized Score

def main(CE, CA, EAG, VE, OE):
    I_CE = get_CE_score(CE)
    I_CA = get_CA_score(CA)
    I_EAG = get_EAG_score(EAG)
    I_VE = get_VE_score2(VE)
    I_OE = get_OE_score(OE)

    return (I_CE, I_CA, I_EAG, I_VE, I_OE)


def get_CE_score(CE):
    if CE < 1.0: return 100 # 0.5
    elif 30 < CE: return 0 # 20

    return 100 - (100 * (CE - 1.0))/29


def get_CA_score(CA):
    if 10 < CA: return 10

    return 100 - (10 * CA)


def get_EAG_score(EAG):
    if EAG < 0.05: return 100
    elif 2.0 < EAG: return 0

    return 100 - (100 * (EAG - 0.05))/1.95


def get_VE_score(VE):
    """
    negative check version
    """
    return VE * 100


def get_VE_score2(VE):
    """
    positive check version
    VE_MED
    """
    if VE < 0.1: return 100
    elif 2.0 < VE: return 0

    return 100 - (100 * (VE - 0.1))/1.9


def get_OE_score(OE):
    return OE * 100



def main_cl(args):
    with open(args.result_json, 'rb') as f:
        tmp = json.load(f)

    for floor, results in tmp.items():
        for key, val in results.items():
            if key == 'CE50': print(F'{key} : {get_CE_score(val)}')
            elif key == 'T-EAG': print(F'{key} : {get_EAG_score(val)}')
            elif key == 'CA_RCS': print(F'{key} : {get_CA_score(val)}')
            elif key == 'Vel_Valid': print(F'{key} : {get_VE_score(val)}')
            elif key == 'Vel_PE50': print(F'{key} : {get_VE_score2(val)}')
            elif key == 'Obstacle': print(F'{key} : {get_OE_score(val)}')
            else:
                print(F'{key} : {val}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--result_json', '-j', type=str)

    args = parser.parse_args()
    main_cl(args)