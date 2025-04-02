#pragma once

#include "mpi_info.h"

void run_pbicgstab(
    double A[][7],
    double x[],
    double b[],
    double r[],
    int sz[3],
    int gc,
    mpi_info mpi
);