#!/bin/bash

#PJM -L rscgrp=b-batch
#PJM -L node=1
#PJM --mpi proc=4
#PJM -L elapse=3:00:00
#PJM -j
#PJM -o windtunnel.log

module load nvidia
module load nvompi

mpiexec -n 4 -display-map bin/windtunnel --clear cases/plain_windtunnel_50
