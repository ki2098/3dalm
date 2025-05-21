#!/usr/bin/env python

import json
import sys
import os
import subprocess

directory = sys.argv[1]

exec = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../bin/2vtk')

with open(os.path.join(directory, 'snapshot.json')) as f:
    ss = json.load(f)
    for s in ss:
        step = s['step']
        fpath = os.path.join(directory, f'inst_{step:010}')
        command = f'{exec} {fpath}'
        print(command)
        subprocess.run([exec, fpath])
        if 'tavg' in s:
            if s['tavg'] == 'yes':
                fpath = os.path.join(directory, f'tavg_{step:010}')
                command = f'{exec} {fpath}'
                print(command)
                subprocess.run([exec, fpath])