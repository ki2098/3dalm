#!/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
import json
import argparse
import math

def get_tavg_U(df: pd.DataFrame):
    row_count = 0
    u_sum = 0.0
    v_sum = 0.0
    w_sum = 0.0
    for row in df.itertuples():
        row_count += 1
        u_sum += row.u
        v_sum += row.v
        w_sum += row.w
        
    return u_sum/row_count, v_sum/row_count, w_sum/row_count

def get_tavg_I(df: pd.DataFrame, U=None):
    if U is None:
        U = get_tavg_U(df)
    Umag = math.sqrt(U[0]**2 + U[1]**2 + U[2]**2)
    
    row_count = 0
    I_sum = 0.0
    for row in df.itertuples():
        row_count += 1
        du = row.u - U[0]
        dv = row.v - U[1]
        dw = row.w - U[2]
        I_sum += math.sqrt((du**2 + dv**2 + dw**2)/3.0)
    
    return I_sum/row_count

parser = argparse.ArgumentParser()

parser.add_argument('case', help='case directory')
parser.add_argument('id', nargs='*', help='probe id, if not given, plot all the monitors')

args = parser.parse_args()
ids = args.id
case_dir = args.case

if not ids:
    with open(f'{case_dir}/setup.json') as f:
        setup = json.load(f)
        ids = list(range(len(setup['probe'])))

for id in ids:
    with open(f'{case_dir}/output/probe{id}.csv') as f:
        comment = f.readline()
        comment = comment[1:].strip()
        df = pd.read_csv(f, comment='#')
        I = get_tavg_I(df, [1,0,0])
        print(f'probe=({comment}), I={I}')
