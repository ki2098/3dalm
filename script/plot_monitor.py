#!/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
import json
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('case', help='case directory')
parser.add_argument('id', nargs='*', help='monitor id, if not given, plot all the monitors')
args = parser.parse_args()
ids = args.id
case_dir = args.case

if not ids:
    with open(f'{case_dir}/setup.json') as f:
        setup = json.load(f)
        ids = list(range(len(setup['monitor'])))

for id in ids:
    with open(f'{case_dir}/output/monitor{id}.csv') as f:
        comment = f.readline()
        comment = comment[1:].strip()
        csv = pd.read_csv(f, comment='#')
        plt.plot(csv['t'], csv['I'], label=comment)

plt.xlabel('t')
plt.ylabel('I')
plt.legend()
plt.grid(True)
plt.show()
