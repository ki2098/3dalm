#include <cstdio>
#include <cmath>
#include "cfd.h"
#include "mv.h"
#include "util.h"

double calc_convection_term(double stencil[], double U[], double D[]) {
    double &valc  = stencil[0];
    double &vale  = stencil[1];
    double &valee = stencil[2];
    double &valw  = stencil[3];
    double &valww = stencil[4];
    double &valn  = stencil[5];
    double &valnn = stencil[6];
    double &vals  = stencil[7];
    double &valss = stencil[8];
    double &valt  = stencil[9];
    double &valtt = stencil[10];
    double &valb  = stencil[11];
    double &valbb = stencil[12];
    double &u     = U[0];
    double &v     = U[1];
    double &w     = U[2];
    double &dx    = D[0];
    double &dy    = D[1];
    double &dz    = D[2];

    double convection = 0;
    convection += u*(- valee + 8*vale - 8*valw + valww)/(12*dx);
    convection += fabs(u)*(valee - 4*vale + 6*valc - 4*valw + valww)/(12*dx);
    convection += v*(- valnn + 8*valn - 8*vals + valss)/(12*dy);
    convection += fabs(v)*(valnn - 4*valn + 6*valc - 4*vals + valss)/(12*dy);
    convection += w*(- valtt + 8*valt - 8*valb + valbb)/(12*dz);
    convection += fabs(w)*(valtt - 4*valt + 6*valc - 4*valb + valbb)/(12*dz);
    return convection;
}

double calc_diffusion_term(double stencil[], double D[], double viscosity) {
    double &valc = stencil[0];
    double &vale = stencil[1];
    double &valw = stencil[2];
    double &valn = stencil[3];
    double &vals = stencil[4];
    double &valt = stencil[5];
    double &valb = stencil[6];
    double txc   = 1./D[0];
    double txe   = 1./D[1];
    double txw   = 1./D[2];
    double tyc   = 1./D[3];
    double tyn   = 1./D[4];
    double tys   = 1./D[5];
    double tzc   = 1./D[6];
    double tzt   = 1./D[7];
    double tzb   = 1./D[8];

    double diffusion = 0;
    diffusion += txc*(txc*(vale - 2*valc + valw) + 0.25*(txe - txw)*(vale - valw));
    diffusion += tyc*(tyc*(valn - 2*valc + vals) + 0.25*(tyn - tys)*(valn - vals));
    diffusion += tzc*(tzc*(valt - 2*valc + valb) + 0.25*(tzt - tzb)*(valt - valb));
    diffusion *= viscosity;
    return diffusion;
}

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
) {
    int cnt = sz[0]*sz[1]*sz[2];

    #pragma acc data present(U[:cnt], Utmp[:cnt], nut[:cnt], dx[:sz[0]], dy[:sz[1]], dz[:sz[2]])
    #pragma acc parallel firstprivate(Re, dt, sz[:3], gc)
    #pragma acc loop independent collapse(3)
    for (int i = gc; i < sz[0] - gc; i ++) {
        for (int j = gc; j < sz[1] - gc; j ++) {
            for (int k = gc; k < sz[2] - gc; k ++) {
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

                double dxyz[] = {dx[i], dy[j], dz[k]};
                double diffusion_stencil_dxyz[] = {
                    dx[i], dx[i + 1], dx[i - 1],
                    dy[j], dy[j + 1], dy[j - 1],
                    dz[k], dz[k + 1], dz[k - 1]
                };
                double viscosity = 1./Re + nut[idc];

                for (int m = 0; m < 3; m ++) {
                    double convection_stencil[] = {
                        Utmp[idc ][m],
                        Utmp[ide ][m],
                        Utmp[idee][m],
                        Utmp[idw ][m],
                        Utmp[idww][m],
                        Utmp[idn ][m],
                        Utmp[idnn][m],
                        Utmp[ids ][m],
                        Utmp[idss][m],
                        Utmp[idt ][m],
                        Utmp[idtt][m],
                        Utmp[idb ][m],
                        Utmp[idbb][m]
                    };
                    double convection = calc_convection_term(convection_stencil, Utmp[idc], dxyz);

                    double diffusion_stencil[] = {
                        Utmp[idc ][m],
                        Utmp[ide ][m],
                        Utmp[idw ][m],
                        Utmp[idn ][m],
                        Utmp[ids ][m],
                        Utmp[idt ][m],
                        Utmp[idb ][m]
                    };
                    double diffusion = calc_diffusion_term(diffusion_stencil, diffusion_stencil_dxyz, viscosity);

                    U[idc][m] = Utmp[idc][m] + dt*(- convection + diffusion);
                }
            }
        }
    }
}

void calc_poisson_rhs(
    double U[][3],
    double rhs[],
    double dt,
    double scale,
    double dx[],
    double dy[],
    double dz[],
    int sz[3],
    int gc,
    mpi_info *mpi
) {
    int cnt = sz[0]*sz[1]*sz[2];

    #pragma acc data present(U[:cnt], rhs[:cnt], dx[:sz[0]], dy[:sz[1]], dz[:sz[2]])
    #pragma acc parallel firstprivate(dt, scale, sz[:3], gc)
    #pragma acc loop independent collapse(3)
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
                divergence += 0.5*(U[ide][0] - U[idw][0])/dx[i];
                divergence += 0.5*(U[idn][1] - U[ids][1])/dy[j];
                divergence += 0.5*(U[idt][2] - U[idb][2])/dz[k];
                rhs[idc] = divergence/(dt*scale);
            }
        }
    }
}

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
) {
    int cnt = sz[0]*sz[1]*sz[2];

    #pragma acc data present(p[:cnt], U[:cnt], dx[:sz[0]], dy[:sz[1]], dz[:sz[2]])
    #pragma acc parallel firstprivate(dt, sz[:3], gc)
    #pragma acc loop independent collapse(3)
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

                double dpdx = 0.5*(p[ide] - p[idw])/dx[i];
                double dpdy = 0.5*(p[idn] - p[ids])/dy[j];
                double dpdz = 0.5*(p[idt] - p[idb])/dz[k];

                U[idc][0] -= dt*dpdx;
                U[idc][1] -= dt*dpdy;
                U[idc][2] -= dt*dpdz;
            }
        }
    }
}

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
) {
    int cnt = sz[0]*sz[1]*sz[2];

    #pragma acc data present(U[:cnt], nut[:cnt], dx[:sz[0]], dy[:sz[1]], dz[:sz[2]])
    #pragma acc parallel firstprivate(Cs, sz[:3], gc)
    #pragma acc loop independent collapse(3)
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

                double dxc = dx[i];
                double dyc = dy[j];
                double dzc = dz[k];
                double volume = dxc*dyc*dzc;

                double dudx = 0.5*(U[ide][0] - U[idw][0])/dxc;
                double dudy = 0.5*(U[idn][0] - U[ids][0])/dyc;
                double dudz = 0.5*(U[idt][0] - U[idb][0])/dzc;
                double dvdx = 0.5*(U[ide][1] - U[idw][1])/dxc;
                double dvdy = 0.5*(U[idn][1] - U[ids][1])/dyc;
                double dvdz = 0.5*(U[idt][1] - U[idb][1])/dzc;
                double dwdx = 0.5*(U[ide][2] - U[idw][2])/dxc;
                double dwdy = 0.5*(U[idn][2] - U[ids][2])/dyc;
                double dwdz = 0.5*(U[idt][2] - U[idb][2])/dzc;

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
    double U[][3],
    double dx[],
    double dy[],
    double dz[],
    int sz[3],
    int gc,
    mpi_info *mpi
) {
    int cnt = sz[0]*sz[1]*sz[2];

    double total = 0;

    #pragma acc data present(U[:cnt], dx[:sz[0]], dy[:sz[1]], dz[:sz[2]])
    #pragma acc parallel firstprivate(sz[:3], gc)
    #pragma acc loop independent reduction(+:total) collapse(3)
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
                divergence += 0.5*(U[ide][0] - U[idw][0])/dx[i];
                divergence += 0.5*(U[idn][1] - U[ids][1])/dy[j];
                divergence += 0.5*(U[idt][2] - U[idb][2])/dz[k];
                total += divergence*divergence;
            }
        }
    }
    
    int effective_cnt = (sz[0] - 2*gc)*(sz[1] - 2*gc)*(sz[2] - 2*gc);
    return sqrt(total/effective_cnt);
}
