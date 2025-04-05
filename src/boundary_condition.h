#pragma once

#include "mpi_info.h"

void apply_velocity_boundary_condition(
    double u[][3],
    double u_previous[][3],
    double u_inflow[3],
    double dt,
    double transform_x[],
    int sz[3],
    int gc,
    mpi_info *mpi
);