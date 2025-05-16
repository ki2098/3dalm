#pragma once

#include <cmath>
#include <mpi.h>
#include "util.h"
#include "mpi_type.h"

#pragma acc routine seq
static Real calc_convection_muscl(Real stencil[13], Real JU[6], Real dxyz[3]) {
    Real fc  = stencil[0];
    Real fe  = stencil[1];
    Real fee = stencil[2];
    Real fw  = stencil[3];
    Real fww = stencil[4];
    Real fn  = stencil[5];
    Real fnn = stencil[6];
    Real fs  = stencil[7];
    Real fss = stencil[8];
    Real ft  = stencil[9];
    Real ftt = stencil[10];
    Real fb  = stencil[11];
    Real fbb = stencil[12];
    Real JUE = JU[0];
    Real JUW = JU[1];
    Real JVN = JU[2];
    Real JVS = JU[3];
    Real JWT = JU[4];
    Real JWB = JU[5];
    Real dx = dxyz[0];
    Real dy = dxyz[1];
    Real dz = dxyz[2];
    Real J = dx*dy*dz;

    Real d1, d2, d3, d4;
    Real s1, s2, s3, s4;
    Real g1, g2, g3, g4, g5, g6;
    const Real k = 1./3.;
    const Real b = (3. - k)/(1. - k);
    const Real e = 1.;

    d4 = fee - fe;
    d3 = fe  - fc;
    d2 = fc  - fw;
    d1 = fw  - fww;
    s4 = sign(d4);
    s3 = sign(d3);
    s2 = sign(d2);
    s1 = sign(d1);
    g6 = s4*fmax(0., fmin(fabs(d4), s4*b*d3));
    g5 = s3*fmax(0., fmin(fabs(d3), s3*b*d4));
    g4 = s3*fmax(0., fmin(fabs(d3), s3*b*d2));
    g3 = s2*fmax(0., fmin(fabs(d2), s2*b*d3));
    g2 = s2*fmax(0., fmin(fabs(d2), s2*b*d1));
    g1 = s1*fmax(0., fmin(fabs(d1), s1*b*d2));
    Real fEr = fe - (e/4.)*((1. - k)*g6 + (1. + k)*g5);
    Real fEl = fc + (e/4.)*((1. - k)*g3 + (1. + k)*g4);
    Real fWr = fc - (e/4.)*((1. - k)*g4 + (1. + k)*g3);
    Real fWl = fw + (e/4.)*((1. - k)*g1 + (1. + k)*g2);
    Real fluxE = 0.5*(JUE*(fEr + fEl) - fabs(JUE)*(fEr - fEl));
    Real fluxW = 0.5*(JUW*(fWr + fWl) - fabs(JUW)*(fWr - fWl));

    d4 = fnn - fn;
    d3 = fn  - fc;
    d2 = fc  - fs;
    d1 = fs  - fss;
    s4 = sign(d4);
    s3 = sign(d3);
    s2 = sign(d2);
    s1 = sign(d1);
    g6 = s4*fmax(0., fmin(fabs(d4), s4*b*d3));
    g5 = s3*fmax(0., fmin(fabs(d3), s3*b*d4));
    g4 = s3*fmax(0., fmin(fabs(d3), s3*b*d2));
    g3 = s2*fmax(0., fmin(fabs(d2), s2*b*d3));
    g2 = s2*fmax(0., fmin(fabs(d2), s2*b*d1));
    g1 = s1*fmax(0., fmin(fabs(d1), s1*b*d2));
    Real fNr = fn - (e/4.)*((1. - k)*g6 + (1. + k)*g5);
    Real fNl = fc + (e/4.)*((1. - k)*g3 + (1. + k)*g4);
    Real fSr = fc - (e/4.)*((1. - k)*g4 + (1. + k)*g3);
    Real fSl = fs + (e/4.)*((1. - k)*g1 + (1. + k)*g2);
    Real fluxN = 0.5*(JVN*(fNr + fNl) - fabs(JVN)*(fNr - fNl));
    Real fluxS = 0.5*(JVS*(fSr + fSl) - fabs(JVS)*(fSr - fSl));

    d4 = ftt - ft;
    d3 = ft  - fc;
    d2 = fc  - fb;
    d1 = fb  - fbb;
    s4 = sign(d4);
    s3 = sign(d3);
    s2 = sign(d2);
    s1 = sign(d1);
    g6 = s4*fmax(0., fmin(fabs(d4), s4*b*d3));
    g5 = s3*fmax(0., fmin(fabs(d3), s3*b*d4));
    g4 = s3*fmax(0., fmin(fabs(d3), s3*b*d2));
    g3 = s2*fmax(0., fmin(fabs(d2), s2*b*d3));
    g2 = s2*fmax(0., fmin(fabs(d2), s2*b*d1));
    g1 = s1*fmax(0., fmin(fabs(d1), s1*b*d2));
    Real fTr = ft - (e/4.)*((1. - k)*g6 + (1. + k)*g5);
    Real fTl = fc + (e/4.)*((1. - k)*g3 + (1. + k)*g4);
    Real fBr = fc - (e/4.)*((1. - k)*g4 + (1. + k)*g3);
    Real fBl = fb + (e/4.)*((1. - k)*g1 + (1. + k)*g2);
    Real fluxT = 0.5*(JWT*(fTr + fTl) - fabs(JWT)*(fTr - fTl));
    Real fluxB = 0.5*(JWB*(fBr + fBl) - fabs(JWB)*(fBr - fBl));

    Real convection = (
        fluxE - fluxW + fluxN - fluxS + fluxT - fluxB
    )/J;

    return convection;
}

#pragma acc routine seq
static Real calc_convection_kk(Real stencil[13], Real U[3], Real dxyz[3]) {
    Real fc  = stencil[0];
    Real fe  = stencil[1];
    Real fee = stencil[2];
    Real fw  = stencil[3];
    Real fww = stencil[4];
    Real fn  = stencil[5];
    Real fnn = stencil[6];
    Real fs  = stencil[7];
    Real fss = stencil[8];
    Real ft  = stencil[9];
    Real ftt = stencil[10];
    Real fb  = stencil[11];
    Real fbb = stencil[12];
    Real u = U[0];
    Real v = U[1];
    Real w = U[2];
    Real dx = dxyz[0];
    Real dy = dxyz[1];
    Real dz = dxyz[2];

    Real convection =
        u*(- fee + 8*fe - 8*fw + fww)/(12*dx) 
    +   fabs(u)*(fee - 4*fe + 6*fc - 4*fw + fww)/(4*dx)
    +   v*(- fnn + 8*fn - 8*fs + fss)/(12*dy) 
    +   fabs(v)*(fnn - 4*fn + 6*fc - 4*fs + fss)/(4*dy)
    +   w*(- ftt + 8*ft - 8*fb + fbb)/(12*dz) 
    +   fabs(w)*(ftt - 4*ft + 6*fc - 4*fb + fbb)/(4*dz);

    return convection;
}

#pragma acc routine seq
static Real calc_diffusion(Real stencil[7], Real xyz[9], Real dxyz[3], Real viscosity) {
    Real valc = stencil[0];
    Real vale = stencil[1];
    Real valw = stencil[2];
    Real valn = stencil[3];
    Real vals = stencil[4];
    Real valt = stencil[5];
    Real valb = stencil[6];
    Real xc = xyz[0];
    Real xe = xyz[1];
    Real xw = xyz[2];
    Real yc = xyz[3];
    Real yn = xyz[4];
    Real ys = xyz[5];
    Real zc = xyz[6];
    Real zt = xyz[7];
    Real zb = xyz[8];
    Real dx = dxyz[0];
    Real dy = dxyz[1];
    Real dz = dxyz[2];

    Real diffusion = viscosity*(
        ((vale - valc)/(xe - xc) - (valc - valw)/(xc - xw))/dx
    +   ((valn - valc)/(yn - yc) - (valc - vals)/(yc - ys))/dy
    +   ((valt - valc)/(zt - zc) - (valc - valb)/(zc - zb))/dz
    );

    return diffusion;
}

static void calc_intermediate_U(
    Real U[][3], Real Uold[][3], Real JU[][3], Real nut[],
    Real x[], Real y[], Real z[],
    Real dx[], Real dy[], Real dz[],
    Real Re, Real dt,
    Int size[3], Int gc,
    MpiInfo *mpi
) {
    MPI_Request req[4];
    const Int thick = 2;
    /** exchange x- */
    if (mpi->rank > 0) {
        Int count = thick*size[1]*size[2];
        Int send_head_id = index(gc        , 0, 0, size);
        Int recv_head_id = index(gc - thick, 0, 0, size);
#pragma acc host_data use_device(Uold)
        MPI_Isend(Uold[send_head_id], 3*count, get_mpi_datatype<Real>(), mpi->rank - 1, 0, MPI_COMM_WORLD, &req[0]);
#pragma acc host_data use_device(Uold)
        MPI_Irecv(Uold[recv_head_id], 3*count, get_mpi_datatype<Real>(), mpi->rank - 1, 0, MPI_COMM_WORLD, &req[1]);
    }
    /** exchange x+ */
    if (mpi->rank < mpi->size - 1) {
        Int count = thick*size[1]*size[2];
        Int send_head_id = index(size[0] - gc - thick, 0, 0, size);
        Int recv_head_id = index(size[0] - gc        , 0, 0, size);
#pragma acc host_data use_device(Uold)
        MPI_Isend(Uold[send_head_id], 3*count, get_mpi_datatype<Real>(), mpi->rank + 1, 0, MPI_COMM_WORLD, &req[2]);
#pragma acc host_data use_device(Uold)
        MPI_Irecv(Uold[recv_head_id], 3*count, get_mpi_datatype<Real>(), mpi->rank + 1, 0, MPI_COMM_WORLD, &req[3]);
    }
    /** wait x- */
    if (mpi->rank > 0) {
        MPI_Wait(&req[0], MPI_STATUS_IGNORE);
        MPI_Wait(&req[1], MPI_STATUS_IGNORE);
    }
    /** wait x+ */
    if (mpi->rank < mpi->size - 1) {
        MPI_Wait(&req[2], MPI_STATUS_IGNORE);
        MPI_Wait(&req[3], MPI_STATUS_IGNORE);
    }

    Int len = size[0]*size[1]*size[2];
#pragma acc kernels loop independent collapse(3) \
present(U[:len], Uold[:len], JU[:len], nut[:len]) \
present(x[:size[0]], y[:size[1]], z[:size[2]]) \
present(dx[:size[0]], dy[:size[1]], dz[:size[2]]) \
copyin(size[:3])
    for (Int i = gc; i < size[0] - gc; i ++) {
    for (Int j = gc; j < size[1] - gc; j ++) {
    for (Int k = gc; k < size[2] - gc; k ++) {
        Int idc  = index(i, j, k, size);
        Int ide  = index(i + 1, j, k, size);
        Int idee = index(i + 2, j, k, size);
        Int idw  = index(i - 1, j, k, size);
        Int idww = index(i - 2, j, k, size);
        Int idn  = index(i, j + 1, k, size);
        Int idnn = index(i, j + 2, k, size);
        Int ids  = index(i, j - 1, k, size);
        Int idss = index(i, j - 2, k, size);
        Int idt  = index(i, j, k + 1, size);
        Int idtt = index(i, j, k + 2, size);
        Int idb  = index(i, j, k - 1, size);
        Int idbb = index(i, j, k - 2, size);
        Real dxyz[] = {dx[i], dy[j], dz[k]};
        Real xyz[] = {
            x[i], x[i + 1], x[i - 1],
            y[j], y[j + 1], y[j - 1],
            z[k], z[k + 1], z[k - 1]
        };
        Real viscosity = 1/Re + nut[idc];
        Real JU_stencil[] = {
            JU[idc][0], JU[idw][0],
            JU[idc][1], JU[ids][1],
            JU[idc][2], JU[idb][2]
        };

#pragma acc loop seq
        for (Int m = 0; m < 3; m ++) {
            Real convection_stencil[] = {
                Uold[idc ][m],
                Uold[ide ][m],
                Uold[idee][m],
                Uold[idw ][m],
                Uold[idww][m],
                Uold[idn ][m],
                Uold[idnn][m],
                Uold[ids ][m],
                Uold[idss][m],
                Uold[idt ][m],
                Uold[idtt][m],
                Uold[idb ][m],
                Uold[idbb][m],
            };
            Real convection = calc_convection_muscl(convection_stencil, JU_stencil, dxyz);

            Real diffusion_stencil[] = {
                Uold[idc ][m],
                Uold[ide ][m],
                Uold[idw ][m],
                Uold[idn ][m],
                Uold[ids ][m],
                Uold[idt ][m],
                Uold[idb ][m],
            };
            Real diffusion = calc_diffusion(diffusion_stencil, xyz, dxyz, viscosity);

            U[idc][m] = Uold[idc][m] + dt*(- convection + diffusion);
        }
    }}}
}

static void interpolate_JU(
    Real U[][3], Real JU[][3],
    Real dx[], Real dy[], Real dz[],
    Int size[3], Int gc,
    MpiInfo *mpi
) {
    MPI_Request req[4];
    const Int thick = 1;
    /** exchange x- */
    if (mpi->rank > 0) {
        Int count = thick*size[1]*size[2];
        Int send_head_id = index(gc        , 0, 0, size);
        Int recv_head_id = index(gc - thick, 0, 0, size);
#pragma acc host_data use_device(U)
        MPI_Isend(U[send_head_id], 3*count, get_mpi_datatype<Real>(), mpi->rank - 1, 0, MPI_COMM_WORLD, &req[0]);
#pragma acc host_data use_device(U)
        MPI_Irecv(U[recv_head_id], 3*count, get_mpi_datatype<Real>(), mpi->rank - 1, 0, MPI_COMM_WORLD, &req[1]);
    }
    /** exchange x+ */
    if (mpi->rank < mpi->size - 1) {
        Int count = thick*size[1]*size[2];
        Int send_head_id = index(size[0] - gc - thick, 0, 0, size);
        Int recv_head_id = index(size[0] - gc        , 0, 0, size);
#pragma acc host_data use_device(U)
        MPI_Isend(U[send_head_id], 3*count, get_mpi_datatype<Real>(), mpi->rank + 1, 0, MPI_COMM_WORLD, &req[2]);
#pragma acc host_data use_device(U)
        MPI_Irecv(U[recv_head_id], 3*count, get_mpi_datatype<Real>(), mpi->rank + 1, 0, MPI_COMM_WORLD, &req[3]);
    }
    /** wait x- */
    if (mpi->rank > 0) {
        MPI_Wait(&req[0], MPI_STATUS_IGNORE);
        MPI_Wait(&req[1], MPI_STATUS_IGNORE);
    }
    /** wait x+ */
    if (mpi->rank < mpi->size - 1) {
        MPI_Wait(&req[2], MPI_STATUS_IGNORE);
        MPI_Wait(&req[3], MPI_STATUS_IGNORE);
    }

    Int len = size[0]*size[1]*size[2];

#pragma acc kernels loop independent collapse(3) \
present(U[:len], JU[:len]) \
present(dx[:size[0]], dy[:size[1]], dz[:size[2]]) \
copyin(size[:3])
    for (Int i = gc - 1; i < size[0] - gc; i ++) {
    for (Int j = gc    ; j < size[1] - gc; j ++) {
    for (Int k = gc    ; k < size[2] - gc; k ++) {
        Int idc = index(i, j, k, size);
        Int ide = index(i + 1, j, k, size);
        Real yz = dy[j]*dz[k];
        Real JUc = U[idc][0]*yz;
        Real JUe = U[ide][0]*yz;
        JU[idc][0] = 0.5*(JUc + JUe);
    }}}

#pragma acc kernels loop independent collapse(3) \
present(U[:len], JU[:len]) \
present(dx[:size[0]], dy[:size[1]], dz[:size[2]]) \
copyin(size[:3])
    for (Int i = gc    ; i < size[0] - gc; i ++) {
    for (Int j = gc - 1; j < size[1] - gc; j ++) {
    for (Int k = gc    ; k < size[2] - gc; k ++) {
        Int idc = index(i, j, k, size);
        Int idn = index(i, j + 1, k, size);
        Real xz = dx[i]*dz[k];
        Real JVc = U[idc][1]*xz;
        Real JVn = U[idn][1]*xz;
        JU[idc][1] = 0.5*(JVc + JVn);
    }}}

#pragma acc kernels loop independent collapse(3) \
present(U[:len], JU[:len]) \
present(dx[:size[0]], dy[:size[1]], dz[:size[2]]) \
copyin(size[:3])
    for (Int i = gc    ; i < size[0] - gc; i ++) {
    for (Int j = gc    ; j < size[1] - gc; j ++) {
    for (Int k = gc - 1; k < size[2] - gc; k ++) {
        Int idc = index(i, j, k, size);
        Int idt = index(i, j, k + 1, size);
        Real xy = dx[i]*dy[j];
        Real JWc = U[idc][2]*xy;
        Real JWt = U[idt][2]*xy;
        JU[idc][2] = 0.5*(JWc + JWt);
    }}}
}

static void calc_poisson_rhs(
    Real JU[][3], Real rhs[],
    Real dx[], Real dy[], Real dz[],
    Real dt, Real scale,
    Int size[3], Int gc,
    MpiInfo *mpi
) {
    Int len = size[0]*size[1]*size[2];
#pragma acc kernels loop independent collapse(3) \
present(JU[:len], rhs[:len]) \
present(dx[:size[0]], dy[:size[1]], dz[:size[2]]) \
copyin(size[:3])
    for (Int i = gc; i < size[0] - gc; i ++) {
    for (Int j = gc; j < size[1] - gc; j ++) {
    for (Int k = gc; k < size[2] - gc; k ++) {
        Int idc = index(i, j, k, size);
        Int idw = index(i - 1, j, k, size);
        Int ids = index(i, j - 1, k, size);
        Int idb = index(i, j, k - 1, size);
        Real JUE = JU[idc][0];
        Real JUW = JU[idw][0];
        Real JVN = JU[idc][1];
        Real JVS = JU[ids][1];
        Real JWT = JU[idc][2];
        Real JWB = JU[idb][2];
        Real J = dx[i]*dy[j]*dz[k];
        Real divergence = (JUE - JUW + JVN - JVS + JWT - JWB)/J;
        rhs[idc] = divergence/(dt*scale);
    }}}
}

static void project_p(
    Real U[][3], Real JU[][3], Real p[],
    Real x[], Real y[], Real z[],
    Real dx[], Real dy[], Real dz[],
    Real dt,
    Int size[3], Int gc,
    MpiInfo *mpi
) {
    MPI_Request req[4];
    const Int thick = 1;
    /** exchange x- */
    if (mpi->rank > 0) {
        Int count = thick*size[1]*size[2];
        Int send_head_id = index(gc        , 0, 0, size);
        Int recv_head_id = index(gc - thick, 0, 0, size);
#pragma acc host_data use_device(p)
        MPI_Isend(&p[send_head_id], count, get_mpi_datatype<Real>(), mpi->rank - 1, 1, MPI_COMM_WORLD, &req[0]);
#pragma acc host_data use_device(p)
        MPI_Irecv(&p[recv_head_id], count, get_mpi_datatype<Real>(), mpi->rank - 1, 1, MPI_COMM_WORLD, &req[1]);
    }
    /** exchange x+ */
    if (mpi->rank < mpi->size - 1) {
        Int count = thick*size[1]*size[2];
        Int send_head_id = index(size[0] - gc - thick, 0, 0, size);
        Int recv_head_id = index(size[0] - gc        , 0, 0, size);
#pragma acc host_data use_device(p)
        MPI_Isend(&p[send_head_id], count, get_mpi_datatype<Real>(), mpi->rank + 1, 1, MPI_COMM_WORLD, &req[2]);
#pragma acc host_data use_device(p)
        MPI_Irecv(&p[recv_head_id], count, get_mpi_datatype<Real>(), mpi->rank + 1, 1, MPI_COMM_WORLD, &req[3]);
    }
    /** wait x- */
    if (mpi->rank > 0) {
        MPI_Wait(&req[0], MPI_STATUS_IGNORE);
        MPI_Wait(&req[1], MPI_STATUS_IGNORE);
    }
    /** wait x+ */
    if (mpi->rank < mpi->size - 1) {
        MPI_Wait(&req[2], MPI_STATUS_IGNORE);
        MPI_Wait(&req[3], MPI_STATUS_IGNORE);
    }
/** FINISHED IN MAY 7 */

    Int len = size[0]*size[1]*size[2];

#pragma acc kernels loop independent collapse(3) \
present(U[:len], p[:len]) \
present(x[:size[0]], y[:size[1]], z[:size[2]]) \
copyin(size[:3])
    for (Int i = gc; i < size[0] - gc; i ++) {
    for (Int j = gc; j < size[1] - gc; j ++) {
    for (Int k = gc; k < size[2] - gc; k ++) {
        Int idc = index(i, j, k, size);
        Int ide = index(i + 1, j, k, size);
        Int idw = index(i - 1, j, k, size);
        Int idn = index(i, j + 1, k, size);
        Int ids = index(i, j - 1, k, size);
        Int idt = index(i, j, k + 1, size);
        Int idb = index(i, j, k - 1, size);
        U[idc][0] -= dt*(p[ide] - p[idw])/(x[i + 1] - x[i - 1]);
        U[idc][1] -= dt*(p[idn] - p[ids])/(y[j + 1] - y[j - 1]);
        U[idc][2] -= dt*(p[idt] - p[idb])/(z[k + 1] - z[k - 1]);
    }}}

#pragma acc kernels loop independent collapse(3) \
present(JU[:len], p[:len]) \
present(x[:size[0]], dy[:size[1]], dz[:size[2]]) \
copyin(size[:3])
    for (Int i = gc - 1; i < size[0] - gc; i ++) {
    for (Int j = gc    ; j < size[1] - gc; j ++) {
    for (Int k = gc    ; k < size[2] - gc; k ++) {
        Int idc = index(i, j, k, size);
        Int ide = index(i + 1, j, k, size);
        Real yz = dy[j]*dz[k];
        Real dpdx = (p[ide] - p[idc])/(x[i + 1] - x[i]);
        JU[idc][0] -= dt*yz*dpdx;
    }}}

#pragma acc kernels loop independent collapse(3) \
present(JU[:len], p[:len]) \
present(dx[:size[0]], y[:size[1]], dz[:size[2]]) \
copyin(size[:3])
    for (Int i = gc    ; i < size[0] - gc; i ++) {
    for (Int j = gc - 1; j < size[1] - gc; j ++) {
    for (Int k = gc    ; k < size[2] - gc; k ++) {
        Int idc = index(i, j, k, size);
        Int idn = index(i, j + 1, k, size);
        Real xz = dx[i]*dz[k];
        Real dpdy = (p[idn] - p[idc])/(y[j + 1] - y[j]);
        JU[idc][1] -= dt*xz*dpdy;
    }}}

#pragma acc kernels loop independent collapse(3) \
present(JU[:len], p[:len]) \
present(dx[:size[0]], dy[:size[1]], z[:size[2]]) \
copyin(size[:3])
    for (Int i = gc    ; i < size[0] - gc; i ++) {
    for (Int j = gc    ; j < size[1] - gc; j ++) {
    for (Int k = gc - 1; k < size[2] - gc; k ++) {
        Int idc = index(i, j, k, size);
        Int idt = index(i, j, k + 1, size);
        Real xy = dx[i]*dy[j];
        Real dpdz = (p[idt] - p[idc])/(z[k + 1] - z[k]);
        JU[idc][2] -= dt*xy*dpdz;
    }}}
}

static void calc_nut(
    Real U[][3], Real nut[],
    Real x[], Real y[], Real z[],
    Real dx[], Real dy[], Real dz[],
    Real Cs,
    Int size[3], Int gc,
    MpiInfo *mpi
) {
    MPI_Request req[4];
    const Int thick = 1;
    /** exchange x- */
    if (mpi->rank > 0) {
        Int count = thick*size[1]*size[2];
        Int send_head_id = index(gc        , 0, 0, size);
        Int recv_head_id = index(gc - thick, 0, 0, size);
#pragma acc host_data use_device(U)
        MPI_Isend(U[send_head_id], 3*count, get_mpi_datatype<Real>(), mpi->rank - 1, 0, MPI_COMM_WORLD, &req[0]);
#pragma acc host_data use_device(U)
        MPI_Irecv(U[recv_head_id], 3*count, get_mpi_datatype<Real>(), mpi->rank - 1, 0, MPI_COMM_WORLD, &req[1]);
    }
    /** exchange x+ */
    if (mpi->rank < mpi->size - 1) {
        Int count = thick*size[1]*size[2];
        Int send_head_id = index(size[0] - gc - thick, 0, 0, size);
        Int recv_head_id = index(size[0] - gc        , 0, 0, size);
#pragma acc host_data use_device(U)
        MPI_Isend(U[send_head_id], 3*count, get_mpi_datatype<Real>(), mpi->rank + 1, 0, MPI_COMM_WORLD, &req[2]);
#pragma acc host_data use_device(U)
        MPI_Irecv(U[recv_head_id], 3*count, get_mpi_datatype<Real>(), mpi->rank + 1, 0, MPI_COMM_WORLD, &req[3]);
    }
    /** wait x- */
    if (mpi->rank > 0) {
        MPI_Wait(&req[0], MPI_STATUS_IGNORE);
        MPI_Wait(&req[1], MPI_STATUS_IGNORE);
    }
    /** wait x+ */
    if (mpi->rank < mpi->size - 1) {
        MPI_Wait(&req[2], MPI_STATUS_IGNORE);
        MPI_Wait(&req[3], MPI_STATUS_IGNORE);
    }

    Int len = size[0]*size[1]*size[2];
#pragma acc kernels loop independent collapse(3) \
present(U[:len], nut[:len]) \
present(x[:size[0]], y[:size[1]], z[:size[2]]) \
present(dx[:size[0]], dy[:size[1]], dz[:size[2]]) \
copyin(size[:3])
    for (Int i = gc; i < size[0] - gc; i ++) {
    for (Int j = gc; j < size[1] - gc; j ++) {
    for (Int k = gc; k < size[2] - gc; k ++) {
        Int idc = index(i, j, k, size);
        Int ide = index(i + 1, j, k, size);
        Int idw = index(i - 1, j, k, size);
        Int idn = index(i, j + 1, k, size);
        Int ids = index(i, j - 1, k, size);
        Int idt = index(i, j, k + 1, size);
        Int idb = index(i, j, k - 1, size);
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

static void calc_q(
    Real U[][3], Real q[],
    Real x[], Real y[], Real z[],
    Int size[3], Int gc,
    MpiInfo *mpi
) {
    MPI_Request req[4];
    const Int thick = 1;
    /** exchange x- */
    if (mpi->rank > 0) {
        Int count = thick*size[1]*size[2];
        Int send_head_id = index(gc        , 0, 0, size);
        Int recv_head_id = index(gc - thick, 0, 0, size);
#pragma acc host_data use_device(U)
        MPI_Isend(U[send_head_id], 3*count, get_mpi_datatype<Real>(), mpi->rank - 1, 0, MPI_COMM_WORLD, &req[0]);
#pragma acc host_data use_device(U)
        MPI_Irecv(U[recv_head_id], 3*count, get_mpi_datatype<Real>(), mpi->rank - 1, 0, MPI_COMM_WORLD, &req[1]);
    }
    /** exchange x+ */
    if (mpi->rank < mpi->size - 1) {
        Int count = thick*size[1]*size[2];
        Int send_head_id = index(size[0] - gc - thick, 0, 0, size);
        Int recv_head_id = index(size[0] - gc        , 0, 0, size);
#pragma acc host_data use_device(U)
        MPI_Isend(U[send_head_id], 3*count, get_mpi_datatype<Real>(), mpi->rank + 1, 0, MPI_COMM_WORLD, &req[2]);
#pragma acc host_data use_device(U)
        MPI_Irecv(U[recv_head_id], 3*count, get_mpi_datatype<Real>(), mpi->rank + 1, 0, MPI_COMM_WORLD, &req[3]);
    }
    /** wait x- */
    if (mpi->rank > 0) {
        MPI_Wait(&req[0], MPI_STATUS_IGNORE);
        MPI_Wait(&req[1], MPI_STATUS_IGNORE);
    }
    /** wait x+ */
    if (mpi->rank < mpi->size - 1) {
        MPI_Wait(&req[2], MPI_STATUS_IGNORE);
        MPI_Wait(&req[3], MPI_STATUS_IGNORE);
    }

    Int len = size[0]*size[1]*size[2];
#pragma acc kernels loop independent collapse(3) \
present(U[:len], q[:len]) \
present(x[:size[0]], y[:size[1]], z[:size[2]]) \
copyin(size[:3])
    for (Int i = gc; i < size[0] - gc; i ++) {
    for (Int j = gc; j < size[1] - gc; j ++) {
    for (Int k = gc; k < size[2] - gc; k ++) {
        Int idc = index(i, j, k, size);
        Int ide = index(i + 1, j, k, size);
        Int idw = index(i - 1, j, k, size);
        Int idn = index(i, j + 1, k, size);
        Int ids = index(i, j - 1, k, size);
        Int idt = index(i, j, k + 1, size);
        Int idb = index(i, j, k - 1, size);
        Real dxew = x[i + 1] - x[i - 1];
        Real dyns = y[j + 1] - y[j - 1];
        Real dztb = z[k + 1] - z[k - 1];
        Real dudx = (U[ide][0] - U[idw][0])/dxew;
        Real dudy = (U[idn][0] - U[ids][0])/dyns;
        Real dudz = (U[idt][0] - U[idb][0])/dztb;
        Real dvdx = (U[ide][1] - U[idw][1])/dxew;
        Real dvdy = (U[idn][1] - U[ids][1])/dyns;
        Real dvdz = (U[idt][1] - U[idb][1])/dztb;
        Real dwdx = (U[ide][2] - U[idw][2])/dxew;
        Real dwdy = (U[idn][2] - U[ids][2])/dyns;
        Real dwdz = (U[idt][2] - U[idb][2])/dztb;
        q[idc] = - 0.5*(dudx*dudx + dvdy*dvdy + dwdz*dwdz + 2*(dudy*dvdx + dudz*dwdx + dvdz*dwdy));
    }}}
}

static void calc_divergence(
    Real JU[][3], Real div[],
    Real dx[], Real dy[], Real dz[],
    Int size[3], Int gc,
    MpiInfo *mpi
) {
    Int len = size[0]*size[1]*size[2];
#pragma acc kernels loop independent collapse(3) \
present(JU[:len], div[:len]) \
present(dx[:size[0]], dy[:size[1]], dz[:size[2]]) \
copyin(size[:3])
    for (Int i = gc; i < size[0] - gc; i ++) {
    for (Int j = gc; j < size[1] - gc; j ++) {
    for (Int k = gc; k < size[2] - gc; k ++) {
        Int idc = index(i, j, k, size);
        Int idw = index(i - 1, j, k, size);
        Int ids = index(i, j - 1, k, size);
        Int idb = index(i, j, k - 1, size);
        Real JUE = JU[idc][0];
        Real JUW = JU[idw][0];
        Real JVN = JU[idc][1];
        Real JVS = JU[ids][1];
        Real JWT = JU[idc][2];
        Real JWB = JU[idb][2];
        Real J = dx[i]*dy[j]*dz[k];
        Real divergence = (JUE - JUW + JVN - JVS + JWT - JWB)/J;
        div[idc] = divergence;
    }}}
}

static Real calc_max_cfl(
    Real U[][3],
    Real dx[], Real dy[], Real dz[],
    Real dt,
    Int size[3], Int gc,
    MpiInfo *mpi
) {
    Real max_cfl = 0;
    Int len = size[0]*size[1]*size[2];
#pragma acc kernels loop independent collapse(3) \
present(U[:len]) \
present(dx[:size[0]], dy[:size[1]], dz[:size[2]]) \
copyin(size[:3]) \
reduction(max:max_cfl)
    for (Int i = gc; i < size[0] - gc; i ++) {
    for (Int j = gc; j < size[1] - gc; j ++) {
    for (Int k = gc; k < size[2] - gc; k ++) {
        Int id = index(i, j, k, size);
        Real local_cfl = 
            fabs(U[id][0])*dt/dx[i] 
        +   fabs(U[id][1])*dt/dy[j]
        +   fabs(U[id][2])*dt/dz[k];
        if (local_cfl > max_cfl) {
            max_cfl = local_cfl;
        }
    }}}

    if (mpi->size > 1) {
        MPI_Allreduce(MPI_IN_PLACE, &max_cfl, 1, get_mpi_datatype<Real>(), MPI_MAX, MPI_COMM_WORLD);
    }

    return max_cfl;
}

static void set_solid_U(
    Real U[][3], Real solid[],
    Int size[3], Int gc
) {
    Int len = size[0]*size[1]*size[2];
#pragma acc kernels loop independent collapse(3) \
present(U[:len], solid[:len]) \
copyin(size[:3])
    for (Int i = gc; i < size[0] - gc; i ++) {
    for (Int j = gc; j < size[1] - gc; j ++) {
    for (Int k = gc; k < size[2] - gc; k ++) {
        Int id = index(i, j, k, size);
#pragma acc loop seq
        for (Int m = 0; m < 3; m ++) {
            U[id][m] = (1. - solid[id])*U[id][m];
        }
    }}}
}