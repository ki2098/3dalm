#!/bin/bash

#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM --mpi proc=1
#PJM -L elapse=3:00:00
#PJM -j
#PJM -o windtunnel_mpi.log

module load nvidia
module load nvompi

mpiexec -n 1 -display-map bin/windtunnel_mpi plain_windtunnel_gk.json
