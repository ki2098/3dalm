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

dx = [0.]*cx
dy = [0.]*cy
dz = [0.]*cz
x = [0.]*cx
y = [0.]*cy
z = [0.]*cz

for i in range(gc, cx - gc):
    dx[i] = xlist[i - gc + 1] - xlist[i - gc]
    x[i]  = xlist[i - gc] + 0.5*dx[i]

for i in range(gc - 1, -1, -1):
    dx[i] = 2*dx[i + 1] - dx[i + 2]
    x[i]  = x[i + 1] - 0.5*(dx[i] + dx[i + 1])

for i in range(cx - gc, cx):
    dx[i] = 2*dx[i - 1] - dx[i - 2]
    x[i]  = x[i - 1] + 0.5*(dx[i] + dx[i - 1])

for j in range(gc, cy - gc):
    dy[j] = ylist[j - gc + 1] - ylist[j - gc]
    y[j]  = ylist[j - gc] + 0.5*dy[j]

for j in range(gc - 1, -1, -1):
    dy[j] = 2*dy[j + 1] - dy[j + 2]
    y[j]  = y[j + 1] - 0.5*(dy[j] + dy[j + 1])

for j in range(cy - gc, cy):
    dy[j] = 2*dy[j - 1] - dy[j - 2]
    y[j]  = y[j - 1] + 0.5*(dy[j] + dy[j - 1])

for k in range(gc, cz - gc):
    dz[k] = zlist[k - gc + 1] - zlist[k - gc]
    z[k]  = zlist[k - gc] + 0.5*dz[k]

for k in range(gc - 1, -1, -1):
    dz[k] = 2*dz[k + 1] - dz[k + 2]
    z[k]  = z[k + 1] - 0.5*(dz[k] + dz[k + 1])

for k in range(cz - gc, cz):
    dz[k] = 2*dz[k - 1] - dz[k - 2]
    z[k]  = z[k - 1] + 0.5*(dz[k] + dz[k - 1])

with open(dst_path, "w") as f:
    f.write(f'{cx} {cy} {cz} {gc}\n')
    for i in range(cx):
        f.write(f'{x[i]:.6}\n')
    for j in range(cy):
        f.write(f'{y[j]:.6}\n')
    for k in range(cz):
        f.write(f'{z[k]:.6}\n')