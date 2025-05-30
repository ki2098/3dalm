#!/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
import json
import argparse
# import tkinter

parser = argparse.ArgumentParser()

parser.add_argument('case', help='case directory')
parser.add_argument('id', nargs='*', help='probe id, if not given, plot all the monitors')
parser.add_argument('--var', help='variable you want to plot')
args = parser.parse_args()
ids = args.id
case_dir = args.case
var = args.var

if not ids:
    with open(f'{case_dir}/setup.json') as f:
        setup = json.load(f)
        ids = list(range(len(setup['probe'])))

# plt.switch_backend('tkagg')


for id in ids:
    with open(f'{case_dir}/output/probe{id}.csv') as f:
        comment = f.readline()
        comment = comment[1:].strip()
        csv = pd.read_csv(f, comment='#')
        row_count = len(csv)
        jump = row_count//1000
        plt.plot(csv['t'][::jump], csv[var][::jump], label=f'probe@({comment})')

plt.xlabel('t')
plt.ylabel('I')
plt.legend()
plt.grid(True)
plt.show()
