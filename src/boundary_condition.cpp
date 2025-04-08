#include "boundary_condition.h"
#include "util.h"

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
) {
    /** x-: inflow */
    for (int i = 0; i < gc; i ++) {
        for (int j = gc; j < sz[1] - gc; j ++) {
            for (int k = gc; k < sz[2] - gc; k ++) {
                for (int m = 0; m < 3; m ++) {
                    U[getid(i, j, k, sz)][m] = Uin[m];
                }
            }
        }
    }

    /** x+: convective outflow */
    for (int i = sz[0] - gc; i < sz[0]; i ++) {
        for (int j = gc; j < sz[1] - gc; j ++) {
            for (int k = gc; k < sz[2] - gc; k ++) {
                int idc  = getid(i    , j, k, sz);
                int idw  = getid(i - 1, j, k, sz);
                int idww = getid(i - 2, j, k, sz);
                for (int m = 0; m < 3; m ++) {
                    double uc  = Uprev[idc ][m];
                    double uw  = Uprev[idw ][m];
                    double uww = Uprev[idww][m];
                    double gradient = 0.5*(3*uc - 4*uw + uww)/dx[i];
                    U[idc][m] = uc - uc*dt*gradient;
                }
            }
        }
    }

    /** y-:  slip */
    for (int i = gc; i < sz[0] - gc; i ++) {
        for (int k = gc; k < sz[2] - gc; k ++) {
            int ji2  = gc + 1;
            int ji1  = gc;
            int jo1  = gc - 1;
            int jo2  = gc - 2;
            int idi2 = getid(i, ji2 , k, sz);
            int idi1 = getid(i, ji1 , k, sz);
            int ido1 = getid(i, jo1, k, sz);
            int ido2 = getid(i, jo2, k, sz);
            double ratioo1 = dy[jo1]/dy[ji1];
            double ratioo2 = (dy[jo1] + 0.5*dy[jo2])/(dy[ji1] + 0.5*dy[ji2]);
            double Ubc[] = {U[idi1][0], 0, U[idi1][2]};
            for (int m = 0; m < 3; m ++) {
                U[ido1][m] = Ubc[m] - ratioo1*(U[idi1][m] - Ubc[m]);
                U[ido2][m] = Ubc[m] - ratioo2*(U[idi2][m] - Ubc[m]);
            }
        }
    }

    /** y+: slip */
    for (int i = gc; i < sz[0] - gc; i ++) {
        for (int k = gc; k < sz[2] - gc; k ++) {
            int ji2  = sz[1] - gc - 2;
            int ji1  = sz[1] - gc - 1;
            int jo1  = sz[1] - gc;
            int jo2  = sz[1] - gc + 1;
            int idi2 = getid(i, ji2 , k, sz);
            int idi1 = getid(i, ji1 , k, sz);
            int ido1 = getid(i, jo1, k, sz);
            int ido2 = getid(i, jo2, k, sz);
            double ratioo1 = dy[jo1]/dy[ji1];
            double ratioo2 = (dy[jo1] + 0.5*dy[jo2])/(dy[ji1] + 0.5*dy[ji2]);
            double Ubc[] = {U[idi1][0], 0, U[idi1][2]};
            for (int m = 0; m < 3; m ++) {
                U[ido1][m] = Ubc[m] - ratioo1*(U[idi1][m] - Ubc[m]);
                U[ido2][m] = Ubc[m] - ratioo2*(U[idi2][m] - Ubc[m]);
            }
        }
    }

    /** z-: non-slip */
    for (int i = gc; i < sz[0] - gc; i ++) {
        for (int j = gc; j < sz[1] - gc; j ++) {
            int ki2 = gc + 1;
            int ki1 = gc;
            int ko1 = gc - 1;
            int ko2 = gc - 2;
            int idi2 = getid(i, j, ki2, sz);
            int idi1 = getid(i, j, ki1, sz);
            int ido1 = getid(i, j, ko1, sz);
            int ido2 = getid(i, j, ko2, sz);
            double ratioo1 = dz[ko1]/dz[ki1];
            double ratioo2 = (dz[ko1] + 0.5*dz[ko2])/(dz[ki1] + 0.5*dz[ki2]);
            double Ubc[] = {0, 0, 0};
            for (int m = 0; m < 3; m ++) {
                U[ido1][m] = Ubc[m] - ratioo1*(U[idi1][m] - Ubc[m]);
                U[ido2][m] = Ubc[m] - ratioo2*(U[idi2][m] - Ubc[m]);
            }
        }
    }

    /** z+: slip */
    for (int i = gc; i < sz[0] - gc; i ++) {
        for (int j = gc; j < sz[1] - gc; j ++) {
            int ki2 = sz[2] - gc - 2;
            int ki1 = sz[2] - gc - 1;
            int ko1 = sz[2] - gc;
            int ko2 = sz[2] - gc + 1;
            int idi2 = getid(i, j, ki2, sz);
            int idi1 = getid(i, j, ki1, sz);
            int ido1 = getid(i, j, ko1, sz);
            int ido2 = getid(i, j, ko2, sz);
            double ratioo1 = dz[ko1]/dz[ki1];
            double ratioo2 = (dz[ko1] + 0.5*dz[ko2])/(dz[ki1] + 0.5*dz[ki2]);
            double Ubc[] = {U[idi1][0], U[idi1][1], 0};
            for (int m = 0; m < 3; m ++) {
                U[ido1][m] = Ubc[m] - ratioo1*(U[idi1][m] - Ubc[m]);
                U[ido2][m] = Ubc[m] - ratioo2*(U[idi2][m] - Ubc[m]);
            }
        }
    }
}

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
) {
    /** x-: gradient = 0 */
    for (int j = gc; j < sz[1] - gc; j ++) {
        for (int k = gc; k < sz[2] - gc; k ++) {
            p[getid(gc - 1, j, k, sz)] = p[getid(gc, j, k, sz)];
        }
    }

    /** x+: fixed value = 0 */
    for (int j = gc; j < sz[1] - gc; j ++) {
        for (int k = gc; k < sz[2] - gc; k ++) {
            p[getid(sz[0] - gc, j, k, sz)] = 0 - p[getid(sz[0] - gc - 1, j, k, sz)];
        }
    }

    /** y-: gradient = 0 */
    for (int i = gc; i < sz[0] - gc; i ++) {
        for (int k = gc; k < sz[2] - gc; k ++) {
            p[getid(i, gc - 1, k, sz)] = p[getid(i, gc, k, sz)];
        }
    }

    /** y+: gradient = 0 */
    for (int i = gc; i < sz[0] - gc; i ++) {
        for (int k = gc; k < sz[2] - gc; k ++) {
            p[getid(i, sz[1] - gc, k, sz)] = p[getid(i, sz[1] - gc - 1, k, sz)];
        }
    }

    /** z-: wall function dp/dz = (1/Re + nut)*(d2w/dz2) */
    for (int i = gc; i < sz[0] - gc; i ++) {
        for (int j = gc; j < sz[1] - gc; j ++) {
            int ktt = gc + 1;
            int kt  = gc;
            int kc  = gc - 1;
            int idtt = getid(i, j, ktt, sz);
            int idt  = getid(i, j, kt , sz);
            int idc  = getid(i, j, kc , sz);
            double viscosity = 1/Re + nut[idt];
            double wwall = 0;
            double wt    = U[idt ][2];
            double wtt   = U[idtt][2];
            double dwdzeta = 2*(wt - wwall);
            double d2wdzeta2 = 4*(2*wwall - 3*wt + wtt)/3;
            double dzwall = z[kt] - z[kc];
            double zzwall = dzwall;
            double zzt    = 1/dz[kt];
            double dzzdzeta = 2*(zzt - zzwall);
            double dpdz = viscosity*(zzwall*zzwall*d2wdzeta2 + zzwall*dzzdzeta*dwdzeta);
            p[getid(i, j, kc, sz)] = p[getid(i, j, kt, sz)] - dzwall*dpdz;
        }
    }

    /** z+: gradient = 0 */
    for (int i = gc; i < sz[0] - gc; i ++) {
        for (int j = gc; j < sz[1] - gc; j ++) {
            p[getid(i, j, sz[2] - gc, sz)] = p[getid(i, j, sz[2] - gc - 1, sz)];
        }
    }
}
