#pragma once

#include "mpi_info.h"

void calc_pseudo_velocity(
    double U[][3],
    double Utmp[][3],
    double nut[],
    double Re,
    double dt,
    double dx[],
    double dy[],
    double dz[],
    int sz[3],
    int gc,
    mpi_info *mpi
);

void calc_poisson_rhs(
    double U[][3],
    double rhs[],
    double dt,
    double dx[],
    double dy[],
    double dz[],
    int sz[3],
    int gc,
    mpi_info *mpi
);

void project_pressure(
    double p[],
    double U[][3],
    double dt,
    double dx[],
    double dy[],
    double dz[],
    int sz[3],
    int gc,
    mpi_info *mpi
);

void calc_eddy_viscosity(
    double U[][3],
    double nut[],
    double Cs,
    double dx[],
    double dy[],
    double dz[],
    int sz[3],
    int gc,
    mpi_info *mpi
);

double monitor_divergence(
    double U[][3],
    double dx[],
    double dy[],
    double dz[],
    int sz[3],
    int gc,
    mpi_info *mpi
);
