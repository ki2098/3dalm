#include <cmath>
#include "json.hpp"
#include "io.h"
#include "util.h"
#include "type.h"

using namespace std;
using json = nlohmann::json;

Real calcConvectionKK(
    Real valc0,
    Real vale1,
    Real vale2,
    Real valw1,
    Real valw2,
    Real valn1,
    Real valn2,
    Real vals1,
    Real vals2,
    Real valt1,
    Real valt2,
    Real valb1,
    Real valb2,
    Real u,
    Real v,
    Real w,
    Real dx,
    Real dy,
    Real dz
) {
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

Real calcDiffusion(
    Real valc,
    Real vale,
    Real valw,
    Real valn,
    Real vals,
    Real valt,
    Real valb,
    Real xc,
    Real xe,
    Real xw,
    Real yc,
    Real yn,
    Real ys,
    Real zc,
    Real zt,
    Real zb,
    Real dx,
    Real dy,
    Real dz,
    Real viscosity
) {
    Real diffusion = 0;

    diffusion += ((vale - valc)/(xe - xc) - (valc - valw)/(xc - xw))/dx;

    diffusion += ((valn - valc)/(yn - yc) - (valc - vals)/(yc - ys))/dy;

    diffusion += ((valt - valc)/(zt - zc) - (valc - valb)/(zc - zb))/dz;

    diffusion *= viscosity;

    return diffusion;
}

void calcPseudoU(
    Real *U,
    Real *Uprev,
    Real *nut,
    Real *x,
    Real *y,
    Real *z,
    Real *dx,
    Real *dy,
    Real *dz,
    Real Re,
    Real dt,
    Int cx,
    Int cy,
    Int cz,
    Int gc,
    MpiInfo &mpi
) {
    Int len = cx*cy*cz;
    for (Int i = gc; i < cx - gc; i ++) {
    for (Int j = gc; j < cy - gc; j ++) {
    for (Int k = gc; k < cz - gc; k ++) {
        Int idc0 = getId(i, j, k, cx, cy, cz);
        Int ide1 = getId(i + 1, j, k, cx, cy, cz);
        Int ide2 = getId(i + 2, j, k, cx, cy, cz);
        Int idw1 = getId(i - 1, j, k, cx, cy, cz);
        Int idw2 = getId(i - 2, j, k, cx, cy, cz);
        Int idn1 = getId(i, j + 1, k, cx, cy, cz);
        Int idn2 = getId(i, j + 2, k, cx, cy, cz);
        Int ids1 = getId(i, j - 1, k, cx, cy, cz);
        Int ids2 = getId(i, j - 2, k, cx, cy, cz);
        Int idt1 = getId(i, j, k + 1, cx, cy, cz);
        Int idt2 = getId(i, j, k + 2, cx, cy, cz);
        Int idb1 = getId(i, j, k - 1, cx, cy, cz);
        Int idb2 = getId(i, j, k - 2, cx, cy, cz);

        Real viscosity = 1/Re + nut[idc0];
        Real dxc = dx[i], dyc = dy[j], dzc = dz[k];
        Real xc = x[i], xe = x[i + 1], xw = x[i - 1];
        Real yc = y[j], yn = y[j + 1], ys = y[j - 1];
        Real zc = z[k], zt = z[k + 1], zb = z[k - 1];
        Real u = (Uprev + 0*len)[idc0];
        Real v = (Uprev + 1*len)[idc0];
        Real w = (Uprev + 2*len)[idc0];

        for (Int m = 0; m < 3; m ++) {
            Real *val = Uprev + m*len;
            Real valc0 = val[idc0];
            Real vale1 = val[ide1];
            Real vale2 = val[ide2];
            Real valw1 = val[idw1];
            Real valw2 = val[idw2];
            Real valn1 = val[idn1];
            Real valn2 = val[idn2];
            Real vals1 = val[ids1];
            Real vals2 = val[ids2];
            Real valt1 = val[idt1];
            Real valt2 = val[idt2];
            Real valb1 = val[idb1];
            Real valb2 = val[idb2];
            Real convection = calcConvectionKK(
                valc0,
                vale1, vale2, valw1, valw2,
                valn1, valn2, vals1, vals2,
                valt1, valt2, valb1, valb2,
                u, v, w,
                dxc, dyc, dzc
            );
            Real diffusion = calcDiffusion(
                valc0,
                vale1, valw1,
                valn1, vals1,
                valt1, valb1,
                xc, xe, xw,
                yc, yn, ys,
                zc, zt, zb,
                dxc, dyc, dzc,
                viscosity
            );

            (U + m*len)[idc0] = valc0 + dt*(- convection + diffusion);
        }
    }}}
}

void calcPoissonRhs(
    Real *U,
    Real *rhs,
    Real *x,
    Real *y,
    Real *z,
    Real dt,
    Real scale,
    Int cx,
    Int cy,
    Int cz,
    Int gc,
    MpiInfo &mpi
) {
    Int len = cx*cy*cz;
    for (Int i = gc; i < cx - gc; i ++) {
    for (Int j = gc; j < cy - gc; j ++) {
    for (Int k = gc; k < cz - gc; k ++) {
        Int idc = getId(i, j, k, cx, cy, cz);
        Int ide = getId(i + 1, j, k, cx, cy, cz);
        Int idw = getId(i - 1, j, k, cx, cy, cz);
        Int idn = getId(i, j + 1, k, cx, cy, cz);
        Int ids = getId(i, j - 1, k, cx, cy, cz);
        Int idt = getId(i, j, k + 1, cx, cy, cz);
        Int idb = getId(i, j, k - 1, cx, cy, cz);
        Real *u = U + 0*len;
        Real *v = U + 1*len;
        Real *w = U + 2*len;
        Real divergence = 0;
        divergence += (u[ide] - u[idw])/(x[i + 1] - x[i - 1]);
        divergence += (v[idn] - v[ids])/(y[j + 1] - y[j - 1]);
        divergence += (w[idt] - w[idb])/(z[k + 1] - z[k - 1]);
        rhs[idc] = divergence/(dt*scale);
    }}}
}

void projectP(
    Real *U,
    Real *p,
    Real *x,
    Real *y,
    Real *z,
    Real dt,
    Int cx,
    Int cy,
    Int cz,
    Int gc,
    MpiInfo &mpi
) {
    Int len = cx*cy*cz;
    for (Int i = gc; i < cx - gc; i ++) {
    for (Int j = gc; j < cy - gc; j ++) {
    for (Int k = gc; k < cz - gc; k ++) {
        Int idc = getId(i, j, k, cx, cy, cz);
        Int ide = getId(i + 1, j, k, cx, cy, cz);
        Int idw = getId(i - 1, j, k, cx, cy, cz);
        Int idn = getId(i, j + 1, k, cx, cy, cz);
        Int ids = getId(i, j - 1, k, cx, cy, cz);
        Int idt = getId(i, j, k + 1, cx, cy, cz);
        Int idb = getId(i, j, k - 1, cx, cy, cz);
        (U + 0*len)[idc] -= dt*(p[ide] - p[idw])/(x[i + 1] - x[i - 1]);
        (U + 1*len)[idc] -= dt*(p[idn] - p[ids])/(y[j + 1] - y[j - 1]);
        (U + 2*len)[idc] -= dt*(p[idt] - p[idb])/(z[k + 1] - z[k - 1]);
    }}}
}

void calcNut(
    Real *U,
    Real *nut,
    Real *x,
    Real *y,
    Real *z,
    Real *dx,
    Real *dy,
    Real *dz,
    Real Cs,
    Int cx,
    Int cy,
    Int cz,
    Int gc,
    MpiInfo &mpi
) {
    Int len = cx*cy*cz;
    for (Int i = gc; i < cx - gc; i ++) {
    for (Int j = gc; j < cy - gc; j ++) {
    for (Int k = gc; k < cz - gc; k ++) {
        Int idc = getId(i, j, k, cx, cy, cz);
        Int ide = getId(i + 1, j, k, cx, cy, cz);
        Int idw = getId(i - 1, j, k, cx, cy, cz);
        Int idn = getId(i, j + 1, k, cx, cy, cz);
        Int ids = getId(i, j - 1, k, cx, cy, cz);
        Int idt = getId(i, j, k + 1, cx, cy, cz);
        Int idb = getId(i, j, k - 1, cx, cy, cz);

        Real dxew = x[i + 1] - x[i - 1];
        Real dyns = y[j + 1] - y[j - 1];
        Real dztb = z[k + 1] - z[k - 1];
        Real volume = dx[i]*dy[j]*dz[k];

        Real *u = U + 0*len;
        Real *v = U + 1*len;
        Real *w = U + 2*len;

        Real dudx = (u[ide] - u[idw])/dxew;
        Real dudy = (u[idn] - u[ids])/dyns;
        Real dudz = (u[idt] - u[idb])/dztb;
        Real dvdx = (v[ide] - v[idw])/dxew;
        Real dvdy = (v[idn] - v[ids])/dyns;
        Real dvdz = (v[idt] - v[idb])/dztb;
        Real dwdx = (w[ide] - w[idw])/dxew;
        Real dwdy = (w[idn] - w[ids])/dyns;
        Real dwdz = (w[idt] - w[idb])/dztb;

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
    Real *U,
    Real *div,
    Real *x,
    Real *y,
    Real *z,
    Int cx,
    Int cy,
    Int cz,
    Int gc,
    MpiInfo &mpi
) {
    Int len = cx*cy*cz;
    for (Int i = gc; i < cx - gc; i ++) {
    for (Int j = gc; j < cy - gc; j ++) {
    for (Int k = gc; k < cz - gc; k ++) {
        Int idc = getId(i, j, k, cx, cy, cz);
        Int ide = getId(i + 1, j, k, cx, cy, cz);
        Int idw = getId(i - 1, j, k, cx, cy, cz);
        Int idn = getId(i, j + 1, k, cx, cy, cz);
        Int ids = getId(i, j - 1, k, cx, cy, cz);
        Int idt = getId(i, j, k + 1, cx, cy, cz);
        Int idb = getId(i, j, k - 1, cx, cy, cz);
        Real *u = U + 0*len;
        Real *v = U + 1*len;
        Real *w = U + 2*len;
        Real divergence = 0;
        divergence += (u[ide] - u[idw])/(x[i + 1] - x[i - 1]);
        divergence += (v[idn] - v[ids])/(y[j + 1] - y[j - 1]);
        divergence += (w[idt] - w[idb])/(z[k + 1] - z[k - 1]);
        div[idc] = divergence;
    }}}
}

Real calcNorm(
    Real *x,
    Int cx,
    Int cy,
    Int cz,
    Int gc,
    MpiInfo &mpi
) {
    Int len = cx*cy*cz;
    Real total = 0;
    for (Int i = gc; i < cx - gc; i ++) {
    for (Int j = gc; j < cy - gc; j ++) {
    for (Int k = gc; k < cz - gc; k ++) {
        total += square(x[getId(i, j, k, cx, cy, cz)]);
    }}}
    return sqrt(total);
}

void calcResidual(
    Real *A,
    Real *x,
    Real *b,
    Real *r,
    Int cx,
    Int cy,
    Int cz,
    Int gc,
    MpiInfo &mpi
) {
    Int len = cx*cy*cz;
    for (Int i = gc; i < cx - gc; i ++) {
    for (Int j = gc; j < cy - gc; j ++) {
    for (Int k = gc; k < cz - gc; k ++) {
        Int idc = getId(i, j, k, cx, cy, cz);
        Int ide = getId(i + 1, j, k, cx, cy, cz);
        Int idw = getId(i - 1, j, k, cx, cy, cz);
        Int idn = getId(i, j + 1, k, cx, cy, cz);
        Int ids = getId(i, j - 1, k, cx, cy, cz);
        Int idt = getId(i, j, k + 1, cx, cy, cz);
        Int idb = getId(i, j, k - 1, cx, cy, cz);

        Real ac = (A + 0*len)[idc];
        Real ae = (A + 1*len)[idc];
        Real aw = (A + 2*len)[idc];
        Real an = (A + 3*len)[idc];
        Real as = (A + 4*len)[idc];
        Real at = (A + 5*len)[idc];
        Real ab = (A + 6*len)[idc];

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
    Real *A,
    Real *x,
    Real *b,
    Real relaxRate,
    Int color,
    Int cx,
    Int cy,
    Int cz,
    Int gc,
    MpiInfo &mpi
) {
    Int len = cx*cy*cz;
    for (Int i = gc; i < cx - gc; i ++) {
    for (Int j = gc; j < cy - gc; j ++) {
    for (Int k = gc; k < cz - gc; k ++) {
        if ((i + j + k)%2 == color) {
            Int idc = getId(i, j, k, cx, cy, cz);
            Int ide = getId(i + 1, j, k, cx, cy, cz);
            Int idw = getId(i - 1, j, k, cx, cy, cz);
            Int idn = getId(i, j + 1, k, cx, cy, cz);
            Int ids = getId(i, j - 1, k, cx, cy, cz);
            Int idt = getId(i, j, k + 1, cx, cy, cz);
            Int idb = getId(i, j, k - 1, cx, cy, cz);

            Real ac = (A + 0*len)[idc];
            Real ae = (A + 1*len)[idc];
            Real aw = (A + 2*len)[idc];
            Real an = (A + 3*len)[idc];
            Real as = (A + 4*len)[idc];
            Real at = (A + 5*len)[idc];
            Real ab = (A + 6*len)[idc];

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