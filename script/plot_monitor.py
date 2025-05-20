#!/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
import sys

filename = sys.argv[1]

with open(filename) as f:
    comment = f.readline()
    print("comment:", comment[1:].lstrip())
    csv = pd.read_csv(f, comment="#")
    plt.plot(csv["t"], csv["I"])
    plt.show()

