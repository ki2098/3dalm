#pragma once

#include "mpi_info.h"

void apply_U_boundary_condition(
    double U[][3],
    double Uprev[][3],
    double Uin[3],
    double dt,
    double dx[],
    double dy[],
    double dz[],
    int sz[3],
    int gc,
    mpi_info *mpi
);

void apply_p_boundary_condition(
    double p[],
    double U[][3],
    double Re,
    double nut[],
    double z[],
    double dz[],
    int sz[3],
    int gc,
    mpi_info *mpi
);