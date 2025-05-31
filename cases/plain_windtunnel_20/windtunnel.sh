#!/bin/bash

#PJM -L rscgrp=b-batch
#PJM -L node=1
#PJM --mpi proc=4
#PJM -L elapse=12:00:00
#PJM -j
#PJM -o run.log

module load nvidia
module load nvompi

date

mpiexec -n 4 -display-map ../../bin/windtunnel --clear .

date