#include <cmath>
#include "json.hpp"
#include "io.h"
#include "util.h"
#include "type.h"

using namespace std;
using json = nlohmann::json;

Real calcConvectionKK(Real13 stencil, Real3 U, Real3 cellSz) {
    Real valc0 = stencil[0];
    Real vale1 = stencil[1];
    Real vale2 = stencil[2];
    Real valw1 = stencil[3];
    Real valw2 = stencil[4];
    Real valn1 = stencil[5];
    Real valn2 = stencil[6];
    Real vals1 = stencil[7];
    Real vals2 = stencil[8];
    Real valt1 = stencil[9];
    Real valt2 = stencil[10];
    Real valb1 = stencil[11];
    Real valb2 = stencil[12];
    Real u = U[0];
    Real v = U[1];
    Real w = U[2];
    Real dx = cellSz[0];
    Real dy = cellSz[1];
    Real dz = cellSz[2];
    const Real a = 0.25;

    Real convection = 0;

    convection += u*(- vale2 + 8*vale1 - 8*valw1 + valw2)/(12*dx);
    convection += a*fabs(u)*(vale2 - 4*vale1 + 6*valc0 - 4*valw1 + valw2)/(dx);

    convection += v*(- valn2 + 8*valn1 - 8*vals1 + vals2)/(12*dy);
    convection += a*fabs(v)*(valn2 - 4*valn1 + 6*valc0 - 4*vals1 + vals2)/(dy);

    convection += w*(- valt2 + 8*valt1 - 8*valb1 + valb2)/(12*dz);
    convection += a*fabs(w)*(valt2 - 4*valt1 + 6*valc0 - 4*valb1 + valb2)/(dz);

    return convection;
}

Real calcDiffusion(Real7 stencil, Real9 coord, Real3 cellSz, Real viscosity) {
    Real valc = stencil[0];
    Real vale = stencil[1];
    Real valw = stencil[2];
    Real valn = stencil[3];
    Real vals = stencil[4];
    Real valt = stencil[5];
    Real valb = stencil[6];
    Real xc = coord[0];
    Real xe = coord[1];
    Real xw = coord[2];
    Real yc = coord[3];
    Real yn = coord[4];
    Real ys = coord[5];
    Real zc = coord[6];
    Real zt = coord[7];
    Real zb = coord[8];
    Real dx = cellSz[0];
    Real dy = cellSz[1];
    Real dz = cellSz[2];

    Real diffusion = 0;

    diffusion += ((vale - valc)/(xe - xc) - (valc - valw)/(xc - xw))/dx;

    diffusion += ((valn - valc)/(yn - yc) - (valc - vals)/(yc - ys))/dy;

    diffusion += ((valt - valc)/(zt - zc) - (valc - valb)/(zc - zb))/dz;

    diffusion *= viscosity;

    return diffusion;
}

void calcPseudoU(
    Real3 U[],
    Real3 UPrev[],
    Real nut[],
    Real Re,
    Real dt,
    Real x[],
    Real y[],
    Real z[],
    Real dx[],
    Real dy[],
    Real dz[],
    Int3 sz,
    Int gc,
    MpiInfo *mpi
) {
    for (Int i = gc; i < sz[0] - gc; i ++) {
    for (Int j = gc; j < sz[1] - gc; j ++) {
    for (Int k = gc; k < sz[2] - gc; k ++) {
        Int idc0 = getId(i, j, k, sz);
        Int ide1 = getId(i + 1, j, k, sz);
        Int ide2 = getId(i + 2, j, k, sz);
        Int idw1 = getId(i - 1, j, k, sz);
        Int idw2 = getId(i - 2, j, k, sz);
        Int idn1 = getId(i, j + 1, k, sz);
        Int idn2 = getId(i, j + 2, k, sz);
        Int ids1 = getId(i, j - 1, k, sz);
        Int ids2 = getId(i, j - 2, k, sz);
        Int idt1 = getId(i, j, k + 1, sz);
        Int idt2 = getId(i, j, k + 2, sz);
        Int idb1 = getId(i, j, k - 1, sz);
        Int idb2 = getId(i, j, k - 2, sz);

        Real3 cellSz = {dx[i], dy[j], dz[k]};
        Real9 coordStencil = {
            x[i], x[i + 1], x[i - 1],
            y[j], y[j + 1], y[j - 1],
            z[k], z[k + 1], z[k - 1]
        };
        Real viscosity = 1/Re + nut[idc0];

        for (Int m = 0; m < 3; m ++) {
            Real13 convectionStencil = {
                UPrev[idc0][m],
                UPrev[ide1][m],
                UPrev[ide2][m],
                UPrev[idw1][m],
                UPrev[idw2][m],
                UPrev[idn1][m],
                UPrev[idn2][m],
                UPrev[ids1][m],
                UPrev[ids2][m],
                UPrev[idt1][m],
                UPrev[idt2][m],
                UPrev[idb1][m],
                UPrev[idb2][m]
            };
            Real convection = calcConvectionKK(convectionStencil, UPrev[idc0], cellSz);

            Real7 diffusionStencil = {
                UPrev[idc0][m],
                UPrev[ide1][m],
                UPrev[idw1][m],
                UPrev[idn1][m],
                UPrev[ids1][m],
                UPrev[idt1][m],
                UPrev[idb1][m],
            };
            Real diffusion = calcDiffusion(diffusionStencil, coordStencil, cellSz, viscosity);

            U[idc0][m] = UPrev[idc0][m] + dt*(- convection + diffusion);
        }
    }}}
}

void calcPoissonRhs(
    Real3 U[],
    Real rhs[],
    Real dt,
    Real scale,
    Real x[],
    Real y[],
    Real z[],
    Int3 sz,
    Int gc,
    MpiInfo *mpi
) {
    for (Int i = gc; i < sz[0] - gc; i ++) {
    for (Int j = gc; j < sz[1] - gc; j ++) {
    for (Int k = gc; k < sz[2] - gc; k ++) {
        Int idc = getId(i, j, k, sz);
        Int ide = getId(i + 1, j, k, sz);
        Int idw = getId(i - 1, j, k, sz);
        Int idn = getId(i, j + 1, k, sz);
        Int ids = getId(i, j - 1, k, sz);
        Int idt = getId(i, j, k + 1, sz);
        Int idb = getId(i, j, k - 1, sz);
        Real divergence = 0;
        divergence += (U[ide][0] - U[idw][0])/(x[i + 1] - x[i - 1]);
        divergence += (U[idn][1] - U[ids][1])/(y[j + 1] - y[j - 1]);
        divergence += (U[idt][2] - U[idb][2])/(z[k + 1] - z[k - 1]);
        rhs[idc] = divergence/(dt*scale);
    }}}
}

void projectP(
    Real p[],
    Real3 U[],
    Real dt,
    Real x[],
    Real y[],
    Real z[],
    Int3 sz,
    Int gc,
    MpiInfo *mpi
) {
    for (Int i = gc; i < sz[0] - gc; i ++) {
    for (Int j = gc; j < sz[1] - gc; j ++) {
    for (Int k = gc; k < sz[2] - gc; k ++) {
        Int idc = getId(i, j, k, sz);
        Int ide = getId(i + 1, j, k, sz);
        Int idw = getId(i - 1, j, k, sz);
        Int idn = getId(i, j + 1, k, sz);
        Int ids = getId(i, j - 1, k, sz);
        Int idt = getId(i, j, k + 1, sz);
        Int idb = getId(i, j, k - 1, sz);
        U[idc][0] -= dt*(p[ide] - p[idw])/(x[i + 1] - x[i - 1]);
        U[idc][1] -= dt*(p[idn] - p[ids])/(y[j + 1] - y[j - 1]);
        U[idc][2] -= dt*(p[idt] - p[idb])/(z[k + 1] - z[k - 1]);
    }}}
}

void calcNut(
    Real3 U[],
    Real nut[],
    Real Cs,
    Real x[],
    Real y[],
    Real z[],
    Real dx[],
    Real dy[],
    Real dz[],
    Int3 sz,
    Int gc,
    MpiInfo *mpi
) {
    for (Int i = gc; i < sz[0] - gc; i ++) {
    for (Int j = gc; j < sz[1] - gc; j ++) {
    for (Int k = gc; k < sz[2] - gc; k ++) {
        Int idc = getId(i, j, k, sz);
        Int ide = getId(i + 1, j, k, sz);
        Int idw = getId(i - 1, j, k, sz);
        Int idn = getId(i, j + 1, k, sz);
        Int ids = getId(i, j - 1, k, sz);
        Int idt = getId(i, j, k + 1, sz);
        Int idb = getId(i, j, k - 1, sz);

        Real dxew = x[i + 1] - x[i - 1];
        Real dyns = y[j + 1] - y[j - 1];
        Real dztb = z[k + 1] - z[k - 1];
        Real volume = dx[i]*dy[j]*dz[k];

        Real dudx = (U[ide][0] - U[idw][0])/dxew;
        Real dudy = (U[idn][0] - U[ids][0])/dyns;
        Real dudz = (U[idt][0] - U[idb][0])/dztb;
        Real dvdx = (U[ide][1] - U[idw][1])/dxew;
        Real dvdy = (U[idn][1] - U[ids][1])/dyns;
        Real dvdz = (U[idt][1] - U[idb][1])/dztb;
        Real dwdx = (U[ide][2] - U[idw][2])/dxew;
        Real dwdy = (U[idn][2] - U[ids][2])/dyns;
        Real dwdz = (U[idt][2] - U[idb][2])/dztb;

        Real s1 = 2*square(dudx);
        Real s2 = 2*square(dvdy);
        Real s3 = 2*square(dwdz);
        Real s4 = square(dudy + dvdx);
        Real s5 = square(dudz + dwdx);
        Real s6 = square(dvdz + dwdy);
        Real shear = sqrt(s1 + s2 + s3 + s4 + s5 + s6);
        Real filter = cbrt(volume);
        nut[idc] = square(Cs*filter)*shear;
    }}}
}

void calcDivergence(
    Real3 U[],
    Real div[],
    Real x[],
    Real y[],
    Real z[],
    Int3 sz,
    Int gc,
    MpiInfo *mpi
) {
    for (Int i = gc; i < sz[0] - gc; i ++) {
    for (Int j = gc; j < sz[1] - gc; j ++) {
    for (Int k = gc; k < sz[2] - gc; k ++) {
        Int idc = getId(i, j, k, sz);
        Int ide = getId(i + 1, j, k, sz);
        Int idw = getId(i - 1, j, k, sz);
        Int idn = getId(i, j + 1, k, sz);
        Int ids = getId(i, j - 1, k, sz);
        Int idt = getId(i, j, k + 1, sz);
        Int idb = getId(i, j, k - 1, sz);

        Real divergence = 0;
        divergence += (U[ide][0] - U[idw][0])/(x[i + 1] - x[i - 1]);
        divergence += (U[idn][1] - U[ids][1])/(y[j + 1] - y[j - 1]);
        divergence += (U[idt][2] - U[idb][2])/(z[k + 1] - z[k - 1]);
        div[idc] = divergence;
    }}}
}

Real calcNorm(
    Real v[],
    Int3 sz,
    Int gc,
    MpiInfo *mpi
) {
    Real total = 0;
    for (Int i = gc; i < sz[0] - gc; i ++) {
    for (Int j = gc; j < sz[1] - gc; j ++) {
    for (Int k = gc; k < sz[2] - gc; k ++) {
        total += square(v[getId(i, j, k, sz)]);
    }}}
    return sqrt(total);
}

void calcResidual(
    Real7 A[],
    Real x[],
    Real b[],
    Real r[],
    Int3 sz,
    Int gc,
    MpiInfo *mpi
) {
    for (Int i = gc; i < sz[0] - gc; i ++) {
    for (Int j = gc; j < sz[1] - gc; j ++) {
    for (Int k = gc; k < sz[2] - gc; k ++) {
        Int idc = getId(i, j, k, sz);
        Int ide = getId(i + 1, j, k, sz);
        Int idw = getId(i - 1, j, k, sz);
        Int idn = getId(i, j + 1, k, sz);
        Int ids = getId(i, j - 1, k, sz);
        Int idt = getId(i, j, k + 1, sz);
        Int idb = getId(i, j, k - 1, sz);

        Real ac = A[idc][0];
        Real ae = A[idc][1];
        Real aw = A[idc][2];
        Real an = A[idc][3];
        Real as = A[idc][4];
        Real at = A[idc][5];
        Real ab = A[idc][6];

        Real xc = x[idc];
        Real xe = x[ide];
        Real xw = x[idw];
        Real xn = x[idn];
        Real xs = x[ids];
        Real xt = x[idt];
        Real xb = x[idb];
        
        r[idc] = b[idc] - (ac*xc + ae*xe + aw*xw + an*xn + as*xs + at*xt + ab*xb);
    }}}
}

void sweepSor(
    Real7 A[],
    Real x[],
    Real b[],
    Real relaxRate,
    Int color,
    Int3 sz,
    Int gc,
    MpiInfo *mpi
) {
    for (Int i = gc; i < sz[0] - gc; i ++) {
    for (Int j = gc; j < sz[1] - gc; j ++) {
    for (Int k = gc; k < sz[2] - gc; k ++) {
        if ((i + j + k)%2 == color) {
            Int idc = getId(i, j, k, sz);
            Int ide = getId(i + 1, j, k, sz);
            Int idw = getId(i - 1, j, k, sz);
            Int idn = getId(i, j + 1, k, sz);
            Int ids = getId(i, j - 1, k, sz);
            Int idt = getId(i, j, k + 1, sz);
            Int idb = getId(i, j, k - 1, sz);

            Real ac = A[idc][0];
            Real ae = A[idc][1];
            Real aw = A[idc][2];
            Real an = A[idc][3];
            Real as = A[idc][4];
            Real at = A[idc][5];
            Real ab = A[idc][6];

            Real xc = x[idc];
            Real xe = x[ide];
            Real xw = x[idw];
            Real xn = x[idn];
            Real xs = x[ids];
            Real xt = x[idt];
            Real xb = x[idb];

            Real relax = (b[idc] - (ac*xc + ae*xe + aw*xw + an*xn + as*xs + at*xt + ab*xb))/ac;

            x[idc] = xc + relaxRate*relax;
        }
    }}}
}

void runSor(
    Real7 A[],
    Real x[],
    Real b[],
    Real r[],
    Real relaxRate,
    Int &iter,
    Int maxIter,
    Real &err,
    Real maxErr,
    Int3 sz,
    Int gc,
    MpiInfo *mpi
) {
    Int effectiveCount = (sz[0] - 2*gc)*(sz[1] - 2*gc)*(sz[2] - 2*gc);
    iter = 0;
    do {
        sweepSor(A, x, b, relaxRate, 0, sz, gc, mpi);
        sweepSor(A, x, b, relaxRate, 1, sz, gc, mpi);
        calcResidual(A, x, b, r, sz, gc, mpi);
        err = calcNorm(r, sz, gc, mpi)/sqrt(effectiveCount);
        iter ++;
    } while (iter < maxIter && err > maxErr);
}

Real prepareA(
    Real7 A[],
    Real x[],
    Real y[],
    Real z[],
    Real dx[],
    Real dy[],
    Real dz[],
    Int3 sz,
    Int gc,
    MpiInfo *mpi
) {
    Real maxDiag = 0;

    for (Int i = gc; i < sz[0] - gc; i ++) {
    for (Int j = gc; j < sz[1] - gc; j ++) {
    for (Int k = gc; k < sz[2] - gc; k ++) {
        Int id = getId(i, j, k, sz);
        Real dxc = dx[i];
        Real dyc = dy[j];
        Real dzc = dz[k];
        Real dxec = x[i + 1] - x[i];
        Real dxcw = x[i] - x[i - 1];
        Real dync = y[j + 1] - y[j];
        Real dycs = y[j] - y[j - 1];
        Real dztc = z[k + 1] - z[k];
        Real dzcb = z[k] - z[k - 1];
        Real ae = 1/(dxc*dxec);
        Real aw = 1/(dxc*dxcw);
        Real an = 1/(dyc*dync);
        Real as = 1/(dyc*dycs);
        Real at = 1/(dzc*dztc);
        Real ab = 1/(dzc*dzcb);
        Real ac = - (ae + aw + an + as + at + ab);
        A[id] = {ac, ae, aw, an, as, at, ab};
        if (fabs(ac) > maxDiag) {
            maxDiag = fabs(ac);
        }
    }}}

    for (Int i = gc; i < sz[0] - gc; i ++) {
    for (Int j = gc; j < sz[1] - gc; j ++) {
    for (Int k = gc; k < sz[2] - gc; k ++) {
        Int id = getId(i, j, k, sz);
        for (Int m = 0; m < 7; m ++) {
            A[id][m] /= maxDiag;
        }
    }}}

    return maxDiag;
}

void applyUBc(
    Real3 U[],
    Real3 UPrev[],
    Real3 UIn,
    Real dt,
    Real x[],
    Real dy[],
    Real dz[],
    Int3 sz,
    Int gc,
    MpiInfo *mpi
) {
    for (Int i = 0 ; i < gc        ; i ++) {
    for (Int j = gc; j < sz[1] - gc; j ++) {
    for (Int k = gc; k < sz[2] - gc; k ++) {
        Int id = getId(i, j, k, sz);
        for (Int m = 0; m < 3; m ++) {
            U[id][m] = UIn[m];
        }
    }}}

    for (Int i = sz[0] - gc; i < sz[0]     ; i ++) {
    for (Int j = gc        ; j < sz[1] - gc; j ++) {
    for (Int k = gc        ; k < sz[2] - gc; k ++) {
        Int id0 = getId(i, j, k, sz);
        Int id1 = getId(i, j, k, sz);
        Int id2 = getId(i, j, k, sz);
        Real h1 = x[i] - x[i - 1];
        Real h2 = x[i] - x[i - 2];
        Real uOut = U[id0][0];
        for (Int m = 0; m < 3; m ++) {
            Real f0 = UPrev[id0][m];
            Real f1 = UPrev[id1][m];
            Real f2 = UPrev[id2][m];
            Real grad = (f0*(h2*h2 - h1*h1) - f1*h2*h2 + f2*h1*h1)/(h1*h2*h2 - h2*h1*h1);
            U[id0][m] = f0 - uOut*dt*grad;
        }
    }}}

    for (Int i = gc; i < sz[0] - gc; i ++) {
    for (Int j = gc; j < sz[1] - gc; j ++) {
        Int ji1 = gc;
        Int ji2 = gc + 1;
        Int jo1 = gc - 1;
        Int jo2 = gc - 2;
    }}
}