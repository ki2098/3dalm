#!/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
import json
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('prefix', help='data file prefix')
parser.add_argument('id', nargs='*', help='monitor id, if not given, plot all the monitors')
args = parser.parse_args()
ids = args.id
prefix = args.prefix

if not ids:
    with open(prefix + '.json') as f:
        setup = json.load(f)
        ids = list(range(len(setup['monitor'])))

for id in ids:
    filename = f'{prefix}_monitor{id}.csv'
    with open(filename) as f:
        comment = f.readline()
        comment = comment[1:].strip()
        csv = pd.read_csv(f, comment='#')
        plt.plot(csv['t'], csv['I'], label=comment)

plt.xlabel('t')
plt.ylabel('I')
plt.legend()
plt.grid(True)
plt.show()
