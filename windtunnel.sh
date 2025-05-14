#!/bin/bash

#PJM -L rscgrp=b-batch
#PJM -L gpu=3
#PJM --mpi proc=3
#PJM -L elapse=3:00:00
#PJM -j
#PJM -o windtunnel.log

module load nvidia
module load nvompi

mpiexec -n 3 -display-map bin/windtunnel plain_windtunnel_gk.json
