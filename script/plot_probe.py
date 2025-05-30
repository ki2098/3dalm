#!/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
import json
import argparse
# import tkinter

parser = argparse.ArgumentParser()

parser.add_argument('case', help='case directory')
parser.add_argument('id', nargs='*', help='probe id, if not given, plot all the monitors')
parser.add_argument('--ax', action='store_true', help='plot time averaged value over probes\' x-coordinate')
args = parser.parse_args()
ids = args.id
case_dir = args.case

if not ids:
    with open(f'{case_dir}/setup.json') as f:
        setup = json.load(f)
        ids = list(range(len(setup['probe'])))

# plt.switch_backend('tkagg')

if args.ax:
    for id in ids:
        with open(f'{case_dir}/output/probe{id}.csv') as f:
            comment = f.readline()
            comment = comment[1:].strip()
            x = comment.split()[0]
            csv = pd.read_csv(f, comment='#')
            avgI = sum(csv['I'])/len(csv['I'])
            plt.scatter(x, avgI)
else:
    for id in ids:
        with open(f'{case_dir}/output/probe{id}.csv') as f:
            comment = f.readline()
            comment = comment[1:].strip()
            csv = pd.read_csv(f, comment='#')
            row_count = len(csv)
            jump = row_count//200
            plt.plot(csv['t'][::jump], csv['I'][::jump], label=f'probe@({comment})')

plt.xlabel('t')
plt.ylabel('I')
plt.legend()
plt.grid(True)
plt.show()
