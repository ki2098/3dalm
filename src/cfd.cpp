#include "cfd.h"
#include "mv.h"
#include "util.h"
#include "cfd_scheme.h"

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
) {
    for (int i = gc; i < sz[0] - gc; i ++) {
        for (int j = gc; j < sz[1] - gc; j ++) {
            for (int k = gc; k < sz[2] - gc; j ++) {
                int idc  = getid(i, j, k, sz);
                int ide  = getid(i + 1, j, k, sz);
                int idee = getid(i + 2, j, k, sz);
                int idw  = getid(i - 1, j, k, sz);
                int idww = getid(i - 2, j, k, sz);
                int idn  = getid(i, j + 1, k, sz);
                int idnn = getid(i, j + 2, k, sz);
                int ids  = getid(i, j - 1, k, sz);
                int idss = getid(i, j - 2, k, sz);
                int idt  = getid(i, j, k + 1, sz);
                int idtt = getid(i, j, k + 2, sz);
                int idb  = getid(i, j, k - 1, sz);
                int idbb = getid(i, j, k - 2, sz);

                double cell_transform[] = {transform_x[i], transform_y[j], transform_z[k]};
                double diffusion_stencil_transform[] = {
                    transform_x[i], transform_x[i + 1], transform_x[i - 1],
                    transform_y[j], transform_y[j + 1], transform_y[j - 1],
                    transform_z[k], transform_z[k + 1], transform_z[k - 1]
                };
                double viscosity = 1./Re + nut[idc];

                for (int m = 0; m < 3; m ++) {
                    double convection_stencil[] = {
                        u_tmp[idc ][m],
                        u_tmp[ide ][m],
                        u_tmp[idee][m],
                        u_tmp[idw ][m],
                        u_tmp[idww][m],
                        u_tmp[idn ][m],
                        u_tmp[idnn][m],
                        u_tmp[ids ][m],
                        u_tmp[idss][m],
                        u_tmp[idt ][m],
                        u_tmp[idtt][m],
                        u_tmp[idb ][m],
                        u_tmp[idbb][m]
                    };
                    double convection = calc_convection_term(convection_stencil, u_tmp[idc], cell_transform);

                    double diffusion_stencil[] = {
                        u_tmp[idc ][m],
                        u_tmp[ide ][m],
                        u_tmp[idw ][m],
                        u_tmp[idn ][m],
                        u_tmp[ids ][m],
                        u_tmp[idt ][m],
                        u_tmp[idb ][m]
                    };
                    double diffusion = calc_diffusion_term(diffusion_stencil, diffusion_stencil_transform, viscosity);

                    u[idc][m] = u_tmp[idc][m] + dt*(- convection + diffusion);
                }
            }
        }
    }
}

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
) {
    for (int i = gc; i < sz[0] - gc; i ++) {
        for (int j = gc; j < sz[1] - gc; j ++) {
            for (int k = gc; k < sz[2] - gc; j ++) {
                int idc = getid(i, j, k, sz);
                int ide = getid(i + 1, j, k, sz);
                int idw = getid(i - 1, j, k, sz);
                int idn = getid(i, j + 1, k, sz);
                int ids = getid(i, j - 1, k, sz);
                int idt = getid(i, j, k + 1, sz);
                int idb = getid(i, j, k - 1, sz);

                double divergence = 0;
                divergence += 0.5*transform_x[i]*(u[ide][0] - u[idw][0]);
                divergence += 0.5*transform_y[j]*(u[idn][1] - u[ids][1]);
                divergence += 0.5*transform_z[k]*(u[idt][2] - u[idb][2]);
                rhs[idc] = divergence/dt;
            }
        }
    }
}

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
) {
    for (int i = gc; i < sz[0] - gc; i ++) {
        for (int j = gc; j < sz[1] - gc; j ++) {
            for (int k = gc; k < sz[2] - gc; j ++) {
                int idc = getid(i, j, k, sz);
                int ide = getid(i + 1, j, k, sz);
                int idw = getid(i - 1, j, k, sz);
                int idn = getid(i, j + 1, k, sz);
                int ids = getid(i, j - 1, k, sz);
                int idt = getid(i, j, k + 1, sz);
                int idb = getid(i, j, k - 1, sz);

                double dpdx = 0.5*transform_x[i]*(p[ide] - p[idw]);
                double dpdy = 0.5*transform_y[j]*(p[idn] - p[ids]);
                double dpdz = 0.5*transform_z[k]*(p[idt] - p[idb]);

                u[idc][0] -= dt*dpdx;
                u[idc][1] -= dt*dpdy;
                u[idc][2] -= dt*dpdz;
            }
        }
    }
}

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
) {
    for (int i = gc; i < sz[0] - gc; i ++) {
        for (int j = gc; j < sz[1] - gc; j ++) {
            for (int k = gc; k < sz[2] - gc; k ++) {
                int idc = getid(i, j, k, sz);
                int ide = getid(i + 1, j, k, sz);
                int idw = getid(i - 1, j, k, sz);
                int idn = getid(i, j + 1, k, sz);
                int ids = getid(i, j - 1, k, sz);
                int idt = getid(i, j, k + 1, sz);
                int idb = getid(i, j, k - 1, sz);

                double tx = transform_x[i];
                double ty = transform_y[j];
                double tz = transform_z[k];
                double volume = 1./(tx*ty*tz);

                double dudx = 0.5*tx*(u[ide][0] - u[idw][0]);
                double dudy = 0.5*ty*(u[idn][0] - u[ids][0]);
                double dudz = 0.5*tz*(u[idt][0] - u[idb][0]);
                double dvdx = 0.5*tx*(u[ide][1] - u[idw][1]);
                double dvdy = 0.5*ty*(u[idn][1] - u[ids][1]);
                double dvdz = 0.5*tz*(u[idt][1] - u[idb][1]);
                double dwdx = 0.5*tx*(u[ide][2] - u[idw][2]);
                double dwdy = 0.5*ty*(u[idn][2] - u[ids][2]);
                double dwdz = 0.5*tz*(u[idt][2] - u[idb][2]);

                double d1 = 2*square(dudx);
                double d2 = 2*square(dvdy);
                double d3 = 2*square(dwdz);
                double d4 = square(dudy + dvdx);
                double d5 = square(dvdz + dwdy);
                double d6 = square(dudz + dwdx);

                double stress_norm = sqrt(d1 + d2 + d3 + d4 + d5 + d6);
                double filter_size = cbrt(volume);
                nut[idc] = square(filter_size*Cs) * stress_norm;
            }
        }
    }
}

double monitor_divergence(
    double u[][3],
    double rhs[],
    double transform_x[],
    double transform_y[],
    double transform_z[],
    int sz[3],
    int gc,
    mpi_info *mpi
) {
    double total = 0;
    for (int i = gc; i < sz[0] - gc; i ++) {
        for (int j = gc; j < sz[1] - gc; j ++) {
            for (int k = gc; k < sz[2] - gc; k ++) {
                int idc = getid(i, j, k, sz);
                int ide = getid(i + 1, j, k, sz);
                int idw = getid(i - 1, j, k, sz);
                int idn = getid(i, j + 1, k, sz);
                int ids = getid(i, j - 1, k, sz);
                int idt = getid(i, j, k + 1, sz);
                int idb = getid(i, j, k - 1, sz);

                double divergence = 0;
                divergence += 0.5*transform_x[i]*(u[ide][0] - u[idw][0]);
                divergence += 0.5*transform_y[j]*(u[idn][1] - u[ids][1]);
                divergence += 0.5*transform_z[k]*(u[idt][2] - u[idb][2]);
                total += divergence*divergence;
            }
        }
    }
    int effective_cnt = (sz[0] - 2*gc)*(sz[1] - 2*gc)*(sz[2] - 2*gc);
    return sqrt(total/effective_cnt);
}
