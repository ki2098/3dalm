#pragma once

#include "mpi_info.h"

void calc_pseudo_velocity(
    double u[][3],
    double u_tmp[][3],
    double nut[],
    double Re,
    double dt,
    double transform_x[],
    double transform_y[],
    double transform_z[],
    int sz[3],
    int gc,
    mpi_info *mpi
);

void calc_poisson_rhs(
    double u[][3],
    double rhs[],
    double dt,
    double transform_x[],
    double transform_y[],
    double transform_z[],
    int sz[3],
    int gc,
    mpi_info *mpi
);

void project_pressure(
    double p[],
    double u[][3],
    double dt,
    double transform_x[],
    double transform_y[],
    double transform_z[],
    int sz[3],
    int gc,
    mpi_info *mpi
);

void calc_eddy_viscosity(
    double u[][3],
    double nut[],
    double Cs,
    double transform_x[],
    double transform_y[],
    double transform_z[],
    int sz[3],
    int gc,
    mpi_info *mpi
);

double monitor_divergence(
    double u[][3],
    double rhs[],
    double transform_x[],
    double transform_y[],
    double transform_z[],
    int sz[3],
    int gc,
    mpi_info *mpi
);
