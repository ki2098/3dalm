#!/usr/bin/env python

import json
import sys
import os
import subprocess

case_dir = sys.argv[1]

exec = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../bin/2vtk')

with open(os.path.join(case_dir, 'output/snapshot.json')) as f:
    ss = json.load(f)
    for s in ss:
        step = s['step']
        fpath = os.path.join(case_dir, f'output/inst_{step:010}')
        command = f'{exec} {fpath}'
        print(command)
        subprocess.run([exec, fpath])
        if 'tavg' in s:
            if s['tavg'] == 'yes':
                fpath = os.path.join(case_dir, f'output/tavg_{step:010}')
                command = f'{exec} {fpath}'
                print(command)
                subprocess.run([exec, fpath])