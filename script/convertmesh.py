#!/usr/bin/env python

import sys
import os

src_dir = sys.argv[1]
dst_path = sys.argv[2]
gc = 2

os.makedirs(os.path.dirname(dst_path), exist_ok=True)

with open(f'{src_dir}/x.txt') as f:
    nx = int(f.readline())
    xlist = []
    for line in f:
        xlist.append(float(line))

with open(f'{src_dir}/y.txt') as f:
    ny = int(f.readline())
    ylist = []
    for line in f:
        ylist.append(float(line))

with open(f'{src_dir}/z.txt') as f:
    nz = int(f.readline())
    zlist = []
    for line in f:
        zlist.append(float(line))

cx = nx - 1 + 2*gc
cy = ny - 1 + 2*gc
cz = nz - 1 + 2*gc
x = [0.]*cx
y = [0.]*cy
z = [0.]*cz

for i in range(gc, cx - gc):
    x[i]  = (xlist[i - gc] + xlist[i - gc + 1])/2

for i in range(gc - 1, -1, -1):
    x[i] = 3*x[i + 1] - 3*x[i + 2] + x[i + 3]

for i in range(cx - gc, cx):
    x[i] = 3*x[i - 1] - 3*x[i - 2] + x[i - 3]

for j in range(gc, cy - gc):
    y[j]  = (ylist[j - gc] + ylist[j - gc + 1])/2

for j in range(gc - 1, -1, -1):
    y[j] = 3*y[j + 1] - 3*y[j + 2] + y[j + 3]

for j in range(cy - gc, cy):
    y[j] = 3*y[j - 1] - 3*y[j - 2] + y[j - 3]

for k in range(gc, cz - gc):
    z[k]  = (zlist[k - gc] + zlist[k - gc + 1])/2

for k in range(gc - 1, -1, -1):
    z[k] = 3*z[k + 1] - 3*z[k + 2] + z[k + 3]

for k in range(cz - gc, cz):
    z[k] = 3*z[k - 1] - 3*z[k - 2] + z[k - 3]

with open(dst_path, "w") as f:
    f.write(f'{cx} {cy} {cz} {gc}\n')
    for i in range(cx):
        f.write(f'{x[i]:.6}\n')
    for j in range(cy):
        f.write(f'{y[j]:.6}\n')
    for k in range(cz):
        f.write(f'{z[k]:.6}\n')