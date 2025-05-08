#!/bin/bash

#PJM -L rscgrp=b-batch
#PJM -L node=1
#PJM --mpi proc=4
#PJM -L elapse=30M
#PJM -j
#PJM -o windtunnel_mpi.log

module load nvidia
module load nvompi

mpiexec -n $PJM_MPI_PROCS -map-by ppr:$PJM_PROC_BY_NODE:node bin/windtunnel_mpi plain_windtunnel.json > windtunnel.log 2>&1
