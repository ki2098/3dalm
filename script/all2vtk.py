#!/usr/bin/env python

import json
import sys
import os
import subprocess

prefix = sys.argv[1]

exec = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../bin/2vtk')

with open(prefix + '_snapshot.json') as f:
    ss = json.load(f)
    for s in ss:
        step = s['step']
        fname = f'{prefix}_{step:010d}'
        command = f'{exec} {fname}'
        print(command)
        subprocess.run([exec, fname])
        if 'tavg' in s:
            if s['tavg'] == 'yes':
                fname = f'{prefix}_tavg_{step:010d}'
                command = f'{exec} {fname}'
                print(command)
                subprocess.run([exec, fname])