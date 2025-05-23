#!/usr/bin/env python

import sys
import os

src_dir = sys.argv[1]
dst_path = sys.argv[2]
gc = 2

os.makedirs(os.path.dirname(dst_path), exist_ok=True)

def build_coordinate(path):
    with open(path) as f:
        nx = int(f.readline())
        xlist = []
        for line in f:
            xlist.append(float(line))

        cx = nx - 1 + 2*gc
        x = [0.]*cx

        for i in range(gc, cx - gc):
            x[i]  = (xlist[i - gc] + xlist[i - gc + 1])/2

        for i in range(gc - 1, -1, -1):
            x[i] = 3*x[i + 1] - 3*x[i + 2] + x[i + 3]

        for i in range(cx - gc, cx):
            x[i] = 3*x[i - 1] - 3*x[i - 2] + x[i - 3]

        return cx, x

cx, x = build_coordinate(f'{src_dir}/x.txt')
cy, y = build_coordinate(f'{src_dir}/y.txt')
cz, z = build_coordinate(f'{src_dir}/z.txt')

with open(dst_path, "w") as f:
    f.write(f'{cx} {cy} {cz} {gc}\n')
    for i in range(cx):
        f.write(f'{x[i]:.6}\n')
    for j in range(cy):
        f.write(f'{y[j]:.6}\n')
    for k in range(cz):
        f.write(f'{z[k]:.6}\n')