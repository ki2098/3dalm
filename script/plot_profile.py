#!/usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import sys

prefix = sys.argv[1]

files = glob.glob(prefix + "*")

plt.switch_backend('tkagg')

for f in files:
    csv = pd.read_csv(f)
    x = csv['Points:0'][0]
    plt.plot(csv['U:0'], csv['Points:2'], label=f'x={x}')

plt.grid(True)
plt.legend()
plt.xlabel('u/Uin')
plt.ylabel('z/R')
plt.show()
