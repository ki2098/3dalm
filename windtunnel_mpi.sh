#!/bin/bash

#PJM -L rscgrp=b-batch
#PJM -L node=1
#PJM -L elapse=00:30:00
#PJM --mpi proc=4
#PJM -j
#PJM -o windtunnel_mpi.log

module load nvidia
module load nvompi

mpiexec -n $PJM_MPI_PROC -map-by ppr:$PJM_PROC_BY_NODE:node bin/windtunnel_mpi plain_windtunnel.json > windtunnel.log 2>&1
