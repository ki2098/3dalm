#pragma once

#include "mpi_info.h"

void run_pbicgstab(
    double A[][7],
    double x[],
    double b[],
    double r[],
    double r0[],
    double p[],
    double q[],
    double s[],
    double phat[],
    double shat[],
    double t[],
    double tmp[],
    double &err,
    double tolerance,
    int &it,
    int max_iteration,
    int sz[3],
    int gc,
    mpi_info *mpi
);