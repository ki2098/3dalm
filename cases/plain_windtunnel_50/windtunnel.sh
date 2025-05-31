#!/bin/bash

#PJM -L rscgrp=b-batch
#PJM -L node=2
#PJM --mpi proc=8
#PJM -L elapse=18:00:00
#PJM -j
#PJM -o run.log

module load nvidia
module load nvompi

date

mpiexec -n 8 -map-by ppr:4:node -display-map ../../bin/windtunnel --clear .

date