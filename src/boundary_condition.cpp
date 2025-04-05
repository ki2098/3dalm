#include "boundary_condition.h"
#include "util.h"

void apply_velocity_boundary_condition(
    double u[][3],
    double u_previous[][3],
    double u_inflow[3],
    double dt,
    double x[],
    double y[],
    double z[],
    double transform_x[],
    double transform_y[],
    double transform_z[],
    int sz[3],
    int gc,
    mpi_info *mpi
) {
    /** xminus: inflow */
    for (int i = 0; i < gc; i ++) {
        for (int j = gc; j < sz[1] - gc; j ++) {
            for (int k = gc; k < sz[2] - gc; k ++) {
                for (int m = 0; m < 3; m ++) {
                    u[getid(i, j, k, sz)][m] = u_inflow[m];
                }
            }
        }
    }

    /** xplus: convective outflow */
    for (int i = sz[0] - gc; i < sz[0]; i ++) {
        for (int j = gc; j < sz[1] - gc; j ++) {
            for (int k = gc; k < sz[2] - gc; k ++) {
                int idc  = getid(i    , j, k, sz);
                int idw  = getid(i - 1, j, k, sz);
                int idww = getid(i - 2, j, k, sz);
                for (int m = 0; m < 3; m ++) {
                    double tx  = transform_x[i];
                    double uc  = u_previous[idc ][m];
                    double uw  = u_previous[idw ][m];
                    double uww = u_previous[idww][m];
                    double gradient = tx*0.5*(3*uc - 4*uw + uww);
                    u[idc][m] = uc - uc*dt*gradient;
                }
            }
        }
    }

    
}