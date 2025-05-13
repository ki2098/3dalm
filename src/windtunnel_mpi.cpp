#include <cmath>
#include <mpi.h>
#include <openacc.h>
#include "io.h"
#include "json.hpp"
#include "mpi_type.h"

using namespace std;
using json = nlohmann::json;

#pragma acc routine seq
Real calc_convection_muscl(Real stencil[13], Real JU[6], Real dxyz[3]) {
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
    g6 = s4*max(0., min(fabs(d4), s4*b*d3));
    g5 = s3*max(0., min(fabs(d3), s3*b*d4));
    g4 = s3*max(0., min(fabs(d3), s3*b*d2));
    g3 = s2*max(0., min(fabs(d2), s2*b*d3));
    g2 = s2*max(0., min(fabs(d2), s2*b*d1));
    g1 = s1*max(0., min(fabs(d1), s1*b*d2));
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
    g6 = s4*max(0., min(fabs(d4), s4*b*d3));
    g5 = s3*max(0., min(fabs(d3), s3*b*d4));
    g4 = s3*max(0., min(fabs(d3), s3*b*d2));
    g3 = s2*max(0., min(fabs(d2), s2*b*d3));
    g2 = s2*max(0., min(fabs(d2), s2*b*d1));
    g1 = s1*max(0., min(fabs(d1), s1*b*d2));
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
    g6 = s4*max(0., min(fabs(d4), s4*b*d3));
    g5 = s3*max(0., min(fabs(d3), s3*b*d4));
    g4 = s3*max(0., min(fabs(d3), s3*b*d2));
    g3 = s2*max(0., min(fabs(d2), s2*b*d3));
    g2 = s2*max(0., min(fabs(d2), s2*b*d1));
    g1 = s1*max(0., min(fabs(d1), s1*b*d2));
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
Real calc_convection_kk(Real stencil[13], Real U[3], Real dxyz[3]) {
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
Real calc_diffusion(Real stencil[7], Real xyz[9], Real dxyz[3], Real viscosity) {
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

void calc_intermediate_U(
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

void interpolate_JU(
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

// void calc_poisson_rhs(
//     Real U[][3], Real rhs[],
//     Real x[], Real y[], Real z[],
//     Real dt, Real scale,
//     Int size[3], Int gc,
//     MpiInfo *mpi
// ) {
//     Int len = size[0]*size[1]*size[2];
// #pragma acc kernels loop independent collapse(3) \
// present(U[:len], rhs[:len]) \
// present(x[:size[0]], y[:size[1]], z[:size[2]]) \
// copyin(size[:3])
//     for (Int i = gc; i < size[0] - gc; i ++) {
//     for (Int j = gc; j < size[1] - gc; j ++) {
//     for (Int k = gc; k < size[2] - gc; k ++) {
//         Int idc = index(i, j, k, size);
//         Int ide = index(i + 1, j, k, size);
//         Int idw = index(i - 1, j, k, size);
//         Int idn = index(i, j + 1, k, size);
//         Int ids = index(i, j - 1, k, size);
//         Int idt = index(i, j, k + 1, size);
//         Int idb = index(i, j, k - 1, size);
//         Real divergence = 
//             (U[ide][0] - U[idw][0])/(x[i + 1] - x[i - 1])
//         +   (U[idn][1] - U[ids][1])/(y[j + 1] - y[j - 1])
//         +   (U[idt][2] - U[idb][2])/(z[k + 1] - z[k - 1]);
//         rhs[idc] = divergence/(dt*scale);
//     }}}
// }

void calc_poisson_rhs(
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

void project_p(
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

void calc_nut(
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

// void calc_divergence(
//     Real U[][3], Real div[],
//     Real x[], Real y[], Real z[],
//     Int size[3], Int gc,
//     MpiInfo *mpi
// ) {
//     Int len = size[0]*size[1]*size[2];
// #pragma acc kernels loop independent collapse(3) \
// present(U[:len], div[:len]) \
// present(x[:size[0]], y[:size[1]], z[:size[2]]) \
// copyin(size[:3])
//     for (Int i = gc; i < size[0] - gc; i ++) {
//     for (Int j = gc; j < size[1] - gc; j ++) {
//     for (Int k = gc; k < size[2] - gc; k ++) {
//         Int idc = index(i, j, k, size);
//         Int ide = index(i + 1, j, k, size);
//         Int idw = index(i - 1, j, k, size);
//         Int idn = index(i, j + 1, k, size);
//         Int ids = index(i, j - 1, k, size);
//         Int idt = index(i, j, k + 1, size);
//         Int idb = index(i, j, k - 1, size);
//         Real divergence = 
//             (U[ide][0] - U[idw][0])/(x[i + 1] - x[i - 1])
//         +   (U[idn][1] - U[ids][1])/(y[j + 1] - y[j - 1])
//         +   (U[idt][2] - U[idb][2])/(z[k + 1] - z[k - 1]);
//         div[idc] = divergence;
//     }}}
// }

void calc_divergence(
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

Real calc_max_cfl(
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

Real calc_l2_norm(
    Real v[],
    Int size[3], Int gc,
    MpiInfo *mpi
) {
    Real total = 0;
    Int len = size[0]*size[1]*size[2];
#pragma acc kernels loop independent collapse(3) \
present(v[:len]) \
copyin(size[:3]) \
reduction(+:total)
    for (Int i = gc; i < size[0] - gc; i ++) {
    for (Int j = gc; j < size[1] - gc; j ++) {
    for (Int k = gc; k < size[2] - gc; k ++) {
        total += square(v[index(i, j, k, size)]);
    }}}

    if (mpi->size > 1) {
        MPI_Allreduce(MPI_IN_PLACE, &total, 1, get_mpi_datatype<Real>(), MPI_SUM, MPI_COMM_WORLD);
    }

    return sqrt(total);
}

void calc_residual(
    Real A[][7], Real x[], Real b[], Real r[],
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
#pragma acc host_data use_device(x)
        MPI_Isend(&x[send_head_id], count, get_mpi_datatype<Real>(), mpi->rank - 1, 2, MPI_COMM_WORLD, &req[0]);
#pragma acc host_data use_device(x)
        MPI_Irecv(&x[recv_head_id], count, get_mpi_datatype<Real>(), mpi->rank - 1, 2, MPI_COMM_WORLD, &req[1]);
    }
    /** exchange x+ */
    if (mpi->rank < mpi->size - 1) {
        Int count = thick*size[1]*size[2];
        Int send_head_id = index(size[0] - gc - thick, 0, 0, size);
        Int recv_head_id = index(size[0] - gc        , 0, 0, size);
#pragma acc host_data use_device(x)
        MPI_Isend(&x[send_head_id], count, get_mpi_datatype<Real>(), mpi->rank + 1, 2, MPI_COMM_WORLD, &req[2]);
#pragma acc host_data use_device(x)
        MPI_Irecv(&x[recv_head_id], count, get_mpi_datatype<Real>(), mpi->rank + 1, 2, MPI_COMM_WORLD, &req[3]);
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
present(A[:len], x[:len], b[:len], r[:len]) \
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

void sweep_sor(
    Real A[][7], Real x[], Real b[],
    Real relax_rate, Int color,
    Int size[3], Int offset[3], Int gc,
    MpiInfo *mpi
) {
    MPI_Request req[4];
    const Int thick = 1;
    /** exchange x- */
    if (mpi->rank > 0) {
        Int count = thick*size[1]*size[2];
        Int send_head_id = index(gc        , 0, 0, size);
        Int recv_head_id = index(gc - thick, 0, 0, size);
#pragma acc host_data use_device(x)
        MPI_Isend(&x[send_head_id], count, get_mpi_datatype<Real>(), mpi->rank - 1, 2, MPI_COMM_WORLD, &req[0]);
#pragma acc host_data use_device(x)
        MPI_Irecv(&x[recv_head_id], count, get_mpi_datatype<Real>(), mpi->rank - 1, 2, MPI_COMM_WORLD, &req[1]);
    }
    /** exchange x+ */
    if (mpi->rank < mpi->size - 1) {
        Int count = thick*size[1]*size[2];
        Int send_head_id = index(size[0] - gc - thick, 0, 0, size);
        Int recv_head_id = index(size[0] - gc        , 0, 0, size);
#pragma acc host_data use_device(x)
        MPI_Isend(&x[send_head_id], count, get_mpi_datatype<Real>(), mpi->rank + 1, 2, MPI_COMM_WORLD, &req[2]);
#pragma acc host_data use_device(x)
        MPI_Irecv(&x[recv_head_id], count, get_mpi_datatype<Real>(), mpi->rank + 1, 2, MPI_COMM_WORLD, &req[3]);
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
present(A[:len], x[:len], b[:len]) \
copyin(size[:3], offset[:3])
    for (Int i = gc; i < size[0] - gc; i ++) {
    for (Int j = gc; j < size[1] - gc; j ++) {
    for (Int k = gc; k < size[2] - gc; k ++) {
    if ((i + j + k + offset[0] + offset[1] + offset[2])%2 == color) {
        Int idc = index(i, j, k, size);
        Int ide = index(i + 1, j, k, size);
        Int idw = index(i - 1, j, k, size);
        Int idn = index(i, j + 1, k, size);
        Int ids = index(i, j - 1, k, size);
        Int idt = index(i, j, k + 1, size);
        Int idb = index(i, j, k - 1, size);

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

        Real relaxation = (b[idc] - (ac*xc + ae*xe + aw*xw + an*xn + as*xs + at*xt + ab*xb))/ac;

        x[idc] = xc + relax_rate*relaxation;
    }}}}
}

void run_sor(
    Real A[][7], Real x[], Real b[], Real r[],
    Real relax_rate, Int &it, Int max_it, Real &err, Real tol,
    Int gsize[3], Int size[3], Int offset[3], Int gc,
    MpiInfo *mpi
) {
    Int effective_count = (gsize[0] - 2*gc)*(gsize[1] - 2*gc)*(gsize[2] - 2*gc);
    it = 0;
    do {
        sweep_sor(A, x, b, relax_rate, 0, size, offset, gc, mpi);
        sweep_sor(A, x, b, relax_rate, 1, size, offset, gc, mpi);
        calc_residual(A, x, b, r, size, gc, mpi);
        err = calc_l2_norm(r, size, gc, mpi)/sqrt(effective_count);
        it ++;
    } while (it < max_it && err > tol);
}

Real build_A(
    Real A[][7],
    Real x[], Real y[], Real z[],
    Real dx[], Real dy[], Real dz[],
    Int size[3], Int gc,
    MpiInfo *mpi
) {
    Real max_diag = 0;
    Int len = size[0]*size[1]*size[2];
#pragma acc kernels loop independent collapse(3) \
present(A[:len]) \
present(x[:size[0]], y[:size[1]], z[:size[2]]) \
present(dx[:size[0]], dy[:size[1]], dz[:size[2]]) \
copyin(size[:3]) \
reduction(max:max_diag)
    for (Int i = gc; i < size[0] - gc; i ++) {
    for (Int j = gc; j < size[1] - gc; j ++) {
    for (Int k = gc; k < size[2] - gc; k ++) {
        Int id = index(i, j, k, size);
        Real dxc = dx[i];
        Real dyc = dy[j];
        Real dzc = dz[k];
        Real dxec = x[i + 1] - x[i    ];
        Real dxcw = x[i    ] - x[i - 1];
        Real dync = y[j + 1] - y[j    ];
        Real dycs = y[j    ] - y[j - 1];
        Real dztc = z[k + 1] - z[k    ];
        Real dzcb = z[k    ] - z[k - 1];
        Real ae = 1/(dxc*dxec);
        Real aw = 1/(dxc*dxcw);
        Real an = 1/(dyc*dync);
        Real as = 1/(dyc*dycs);
        Real at = 1/(dzc*dztc);
        Real ab = 1/(dzc*dzcb);
        Real ac = - (ae + aw + an + as + at + ab);
        A[id][0] = ac;
        A[id][1] = ae;
        A[id][2] = aw;
        A[id][3] = an;
        A[id][4] = as;
        A[id][5] = at;
        A[id][6] = ab;
        if (fabs(ac) > max_diag) {
            max_diag = fabs(ac);
        }
    }}}

    if (mpi->size > 1) {
        MPI_Allreduce(MPI_IN_PLACE, &max_diag, 1, get_mpi_datatype<Real>(), MPI_MAX, MPI_COMM_WORLD);
    }

#pragma acc kernels loop independent collapse(3) \
present(A[:len]) \
copyin(size[:3])
    for (Int i = gc; i < size[0] - gc; i ++) {
    for (Int j = gc; j < size[1] - gc; j ++) {
    for (Int k = gc; k < size[2] - gc; k ++) {
        Int id = index(i, j, k, size);
#pragma acc loop seq
        for (Int m = 0; m < 7; m ++) {
            A[id][m] /= max_diag;
        }
    }}}

    return max_diag;
}

void apply_Ubc(
    Real U[][3], Real Uold[][3], Real Uin[3],
    Real dx[], Real dy[], Real dz[],
    Real dt,
    Int size[3], Int gc,
    MpiInfo *mpi
) {
    Int len = size[0]*size[1]*size[2];

    /** x- fixed value inflow */
    if (mpi->rank == 0) {
#pragma acc kernels loop independent collapse(3) \
present(U[:len]) \
copyin(Uin[:3]) \
copyin(size[:3])
        for (Int i = 0; i < gc; i ++) {
        for (Int j = gc; j < size[1] - gc; j ++) {
        for (Int k = gc; k < size[2] - gc; k ++) {
            Int id = index(i, j, k, size);
            U[id][0] = Uin[0];
            U[id][1] = Uin[1];
            U[id][2] = Uin[2];
        }}}
    }

    /** x+ convective outflow */
    if (mpi->rank == mpi->size - 1) {
#pragma acc kernels loop independent collapse(3) \
present(U[:len], Uold[:len]) \
present(dx[:size[0]], dy[:size[1]], dz[:size[2]]) \
copyin(size[:3])
        for (Int i = size[0] - gc; i < size[0]; i ++) {
        for (Int j = gc; j < size[1] - gc; j ++) {
        for (Int k = gc; k < size[2] - gc; k ++) {
            Int id0 = index(i    , j, k, size);
            Int id1 = index(i - 1, j, k, size);
            Int id2 = index(i - 2, j, k, size);
            Real uout = U[id0][0];
            for (Int m = 0; m < 3; m ++) {
                Real f0 = Uold[id0][m];
                Real f1 = Uold[id1][m];
                Real f2 = Uold[id2][m];
                Real grad = (3*f0 - 4*f1 + f2)/2;
                U[id0][m] = f0 - uout*dt*grad/dx[i];
            }
        }}}
    }

    /** y- slip */
#pragma acc kernels loop independent collapse(2) \
present(U[:len]) \
present(dx[:size[0]], dy[:size[1]], dz[:size[2]]) \
copyin(size[:3])
    for (Int i = gc; i < size[0] - gc; i ++) {
    for (Int k = gc; k < size[2] - gc; k ++) {
        Int ji  = gc;
        Int jii = gc + 1;
        Int jo  = gc - 1;
        Int joo = gc - 2;
        Int idi  = index(i, ji , k, size);
        Int idii = index(i, jii, k, size);
        Int ido  = index(i, jo , k, size);
        Int idoo = index(i, joo, k, size);
        Real hi  = 0.5*dy[ji ];
        Real hii = 0.5*dy[jii] + dy[ji];
        Real ho  = 0.5*dy[jo ];
        Real hoo = 0.5*dy[joo] + dy[jo];
        Real Ubc[] = {U[idi][0], 0, U[idi][2]};
        for (Int m = 0; m < 3; m ++) {
            U[ido ][m] = Ubc[m] - (U[idi ][m] - Ubc[m])*(ho /hi );
            U[idoo][m] = Ubc[m] - (U[idii][m] - Ubc[m])*(hoo/hii);
        }
    }}

    /** y+ slip */
#pragma acc kernels loop independent collapse(2) \
present(U[:len]) \
present(dx[:size[0]], dy[:size[1]], dz[:size[2]]) \
copyin(size[:3])
    for (Int i = gc; i < size[0] - gc; i ++) {
    for (Int k = gc; k < size[2] - gc; k ++) {
        Int ji  = size[1] - gc - 1;
        Int jii = size[1] - gc - 2;
        Int jo  = size[1] - gc;
        Int joo = size[1] - gc + 1;
        Int idi  = index(i, ji , k, size);
        Int idii = index(i, jii, k, size);
        Int ido  = index(i, jo , k, size);
        Int idoo = index(i, joo, k, size);
        Real hi  = 0.5*dy[ji ];
        Real hii = 0.5*dy[jii] + dy[ji];
        Real ho  = 0.5*dy[jo ];
        Real hoo = 0.5*dy[joo] + dy[jo];
        Real Ubc[] = {U[idi][0], 0, U[idi][2]};
        for (Int m = 0; m < 3; m ++) {
            U[ido ][m] = Ubc[m] - (U[idi ][m] - Ubc[m])*(ho /hi );
            U[idoo][m] = Ubc[m] - (U[idii][m] - Ubc[m])*(hoo/hii);
        }
    }}

    /** z- non slip */
#pragma acc kernels loop independent collapse(2) \
present(U[:len]) \
present(dx[:size[0]], dy[:size[1]], dz[:size[2]]) \
copyin(size[:3])
    for (Int i = gc; i < size[0] - gc; i ++) {
    for (Int j = gc; j < size[1] - gc; j ++) {
        Int ki  = gc;
        Int kii = gc + 1;
        Int ko  = gc - 1;
        Int koo = gc - 2;
        Int idi  = index(i, j, ki , size);
        Int idii = index(i, j, kii, size);
        Int ido  = index(i, j, ko , size);
        Int idoo = index(i, j, koo, size);
        Real hi  = 0.5*dz[ki ];
        Real hii = 0.5*dz[kii] + dz[ki];
        Real ho  = 0.5*dz[ko ];
        Real hoo = 0.5*dz[koo] + dz[ko];
        Real Ubc[] = {0, 0, 0};
        for (Int m = 0; m < 3; m ++) {
            U[ido ][m] = Ubc[m] - (U[idi ][m] - Ubc[m])*(ho /hi );
            U[idoo][m] = Ubc[m] - (U[idii][m] - Ubc[m])*(hoo/hii);
        }
    }}

    /** z+ slip */
#pragma acc kernels loop independent collapse(2) \
present(U[:len]) \
present(dx[:size[0]], dy[:size[1]], dz[:size[2]]) \
copyin(size[:3])
    for (Int i = gc; i < size[0] - gc; i ++) {
    for (Int j = gc; j < size[1] - gc; j ++) {
        Int ki  = size[2] - gc - 1;
        Int kii = size[2] - gc - 2;
        Int ko  = size[2] - gc;
        Int koo = size[2] - gc + 1;
        Int idi  = index(i, j, ki , size);
        Int idii = index(i, j, kii, size);
        Int ido  = index(i, j, ko , size);
        Int idoo = index(i, j, koo, size);
        Real hi  = 0.5*dz[ki ];
        Real hii = 0.5*dz[kii] + dz[ki];
        Real ho  = 0.5*dz[ko ];
        Real hoo = 0.5*dz[koo] + dz[ko];
        Real Ubc[] = {U[idi][0], U[idi][1], 0};
        for (Int m = 0; m < 3; m ++) {
            U[ido ][m] = Ubc[m] - (U[idi ][m] - Ubc[m])*(ho /hi );
            U[idoo][m] = Ubc[m] - (U[idii][m] - Ubc[m])*(hoo/hii);
        }
    }}
}

void apply_JUbc(
    Real JU[][3], Real Uin[3],
    Real dy[], Real dz[],
    Int size[3], Int gc,
    MpiInfo *mpi
) {
    Int len = size[0]*size[1]*size[2];

    /** x- fixed value inflow */
    if (mpi->rank == 0) {
#pragma acc kernels loop independent collapse(2) \
present(JU[:len]) \
present(dy[:size[1]], dz[:size[2]]) \
copyin(Uin[:3]) \
copyin(size[:3])
        for (Int j = gc; j < size[1] - gc; j ++) {
        for (Int k = gc; k < size[2] - gc; k ++) {
            Int i = gc - 1;
            Int id = index(i, j, k, size);
            Real yz = dy[j]*dz[k];
            JU[id][0] = yz*Uin[0];
        }}
    }
    
    /** y- JV = 0 */
#pragma acc kernels loop independent collapse(2) \
present(JU[:len]) \
copyin(size[:3])
    for (Int i = gc; i < size[0] - gc; i ++) {
    for (Int k = gc; k < size[2] - gc; k ++) {
        Int j = gc - 1;
        Int id = index(i, j, k, size);
        JU[id][1] = 0;
    }}

    /** y+ JV = 0 */
#pragma acc kernels loop independent collapse(2) \
present(JU[:len]) \
copyin(size[:3])
    for (Int i = gc; i < size[0] - gc; i ++) {
    for (Int k = gc; k < size[2] - gc; k ++) {
        Int j = size[1] - gc - 1;
        Int id = index(i, j, k, size);
        JU[id][1] = 0;
    }}

    /** z- JW = 0 */
#pragma acc kernels loop independent collapse(2) \
present(JU[:len]) \
copyin(size[:3])
    for (Int i = gc; i < size[0] - gc; i ++) {
    for (Int j = gc; j < size[1] - gc; j ++) {
        Int k = gc - 1;
        Int id = index(i, j, k, size);
        JU[id][2] = 0;
    }}

    /** z+ JW = 0 */
#pragma acc kernels loop independent collapse(2) \
present(JU[:len]) \
copyin(size[:3])
    for (Int i = gc; i < size[0] - gc; i ++) {
    for (Int j = gc; j < size[1] - gc; j ++) {
        Int k = size[2] - gc - 1;
        Int id = index(i, j, k, size);
        JU[id][2] = 0;
    }}
}

void apply_pbc(
    Real U[][3], Real p[], Real nut[],
    Real z[], Real dx[], Real dy[], Real dz[],
    Real Re,
    Int size[3], Int gc,
    MpiInfo *mpi
) {
    Int len = size[0]*size[1]*size[2];

    /** x- grad = 0 */
    if (mpi->rank == 0) {
#pragma acc kernels loop independent collapse(2) \
present(p[:len]) \
copyin(size[:3])
        for (Int j = gc; j < size[1] - gc; j ++) {
        for (Int k = gc; k < size[2] - gc; k ++) {
            Int ii = gc;
            Int io = gc - 1;
            p[index(io, j, k, size)] = p[index(ii, j, k, size)];
        }}
        // printf("x-\n");
    }

    /** x+ value = 0 */
    if (mpi->rank == mpi->size - 1) {
#pragma acc kernels loop independent collapse(2) \
present(p[:len]) \
present(dx[:size[0]], dy[:size[1]], dz[:size[2]]) \
copyin(size[:3])
        for (Int j = gc; j < size[1] - gc; j ++) {
        for (Int k = gc; k < size[2] - gc; k ++) {
            Int ii = size[0] - gc - 1;
            Int io = size[0] - gc;
            Real hi = 0.5*dx[ii];
            Real ho = 0.5*dx[io];
            Real pbc = 0;
            p[index(io, j, k, size)] = pbc - (p[index(ii, j, k, size)] - pbc)*(ho/hi);
            // printf("%ld %ld %lf\n", j, k, hi);
        }}
        // printf("x+\n");
    }

    /** y- grad = 0 */
#pragma acc kernels loop independent collapse(2) \
present(p[:len]) \
copyin(size[:3])
    for (Int i = gc; i < size[0] - gc; i ++) {
    for (Int k = gc; k < size[2] - gc; k ++) {
        Int ji = gc;
        Int jo = gc - 1;
        p[index(i, jo, k, size)] = p[index(i, ji, k, size)];
    }}
    // printf("y-\n");

    /** y+ grad = 0 */
#pragma acc kernels loop independent collapse(2) \
present(p[:len]) \
copyin(size[:3])
    for (Int i = gc; i < size[0] - gc; i ++) {
    for (Int k = gc; k < size[2] - gc; k ++) {
        Int ji = size[1] - gc - 1;
        Int jo = size[1] - gc;
        p[index(i, jo, k, size)] = p[index(i, ji, k, size)];
    }}
    // printf("y+\n");

    /** z- wall function dp/dz = (1/Re + nut)ddw/dzz */
#pragma acc kernels loop independent collapse(2) \
present(U[:len], p[:len], nut[:len]) \
present(z[:size[2]]) \
present(dx[:size[0]], dy[:size[1]], dz[:size[2]]) \
copyin(size[:3])
    for (Int i = gc; i < size[0] - gc; i ++) {
    for (Int j = gc; j < size[1] - gc; j ++) {
        Int ki  = gc;
        Int kii = gc + 1;
        Int ko  = gc - 1;
        Int idi  = index(i, j, ki , size);
        Int idii = index(i, j, kii, size);
        Int ido  = index(i, j, ko , size);
        Real w0 = 0;
        Real w1 = U[idi ][2];
        Real w2 = U[idii][2];
        Real h1 = 0.5*dz[ki ];
        Real h2 = 0.5*dz[kii] + dz[ki];
        Real ddwdzz = 
            (2*(h1 - h2)*w0 + 2*h2*w1 - 2*h1*w2)
            /(h1*h1*h2 - h1*h2*h2);
        Real dpdz = (1/Re + nut[idi])*ddwdzz;
        p[ido] = p[idi] - dpdz*(z[ki] - z[ko]);
    }}
    // printf("z-\n");

    /** z+ grad = 0 */
#pragma acc kernels loop independent collapse(2) \
present(p[:len]) \
copyin(size[:3])
    for (Int i = gc; i < size[0] - gc; i ++) {
    for (Int j = gc; j < size[1] - gc; j ++) {
        Int ki = size[2] - gc - 1;
        Int ko = size[2] - gc;
        p[index(i, j, ko, size)] = p[index(i, j, ki, size)];
    }}
    // printf("z+\n");
}

void calc_partition(
    Int gsize[3], Int size[3], Int offset[3], Int gc,
    MpiInfo *mpi
) {
    size[1] = gsize[1];
    size[2] = gsize[2];
    offset[1] = 0;
    offset[2] = 0;

    Int inner_x_count = gsize[0] - 2*gc;
    Int segment_len = inner_x_count/mpi->size;
    Int leftover = inner_x_count%mpi->size;

    size[0] = segment_len + 2*gc;
    if (mpi->rank < leftover) {
        size[0] ++;
    }
    offset[0] = segment_len*mpi->rank;
    if (mpi->rank < leftover) {
        offset[0] += mpi->rank;
    } else {
        offset[0] += leftover;
    }
}

struct Runtime {
    Int step = 0, max_step;
    Real dt;

    Real get_time() {
        return dt*step;
    }

    void initialize(Int max_step, Real dt) {
        this->max_step = max_step;
        this->dt = dt;

        // printf("RUNTIME INFO\n");
        // printf("\tmax step = %ld\n", this->max_step);
        // printf("\tdt = %lf\n", this->dt);
    }
};

struct Mesh {
    Real *x, *y, *z, *dx, *dy, *dz;

    void initialize_from_path(string path, Int size[3], Int gc, MpiInfo *mpi) {
        build_mesh(path, x, y, z, dx, dy, dz, size, gc, mpi);

#pragma acc enter data \
copyin(x[:size[0]], y[:size[1]], z[:size[2]]) \
copyin(dx[:size[0]], dy[:size[1]], dz[:size[2]])

        // printf("MESH INFO\n");
        // printf("\tfolder = %s\n", path.c_str());
        // printf("\tsize = (%ld %ld %ld)\n", size[0], size[1], size[2]);
        // printf("\tguide cell = %ld\n", gc);
    }

    void initialize_from_global_mesh(Mesh *gmesh, Int gsize[3], Int size[3], Int offset[3], Int gc, MpiInfo *mpi) {
        // size[1] = gsize[1];
        // size[2] = gsize[2];
        // offset[1] = 0;
        // offset[2] = 0;

        // Int inner_x_count = gsize[0] - 2*gc;
        // Int segment_len = inner_x_count/mpi->size;
        // Int leftover = inner_x_count%mpi->size;

        // size[0] = segment_len + 2*gc;
        // if (mpi->rank < leftover) {
        //     size[0] ++;
        // }
        // offset[0] = segment_len*mpi->rank;
        // if (mpi->rank < leftover) {
        //     offset[0] += mpi->rank;
        // } else {
        //     offset[0] += leftover;
        // }
        calc_partition(gsize, size, offset, gc, mpi);

        x = new Real[size[0]];
        dx = new Real[size[0]];
        for (Int i = 0; i < size[0]; i ++) {
            x[i] = gmesh->x[i + offset[0]];
            dx[i] = gmesh->dx[i + offset[0]];
        }

        y = new Real[size[1]];
        dy = new Real[size[1]];
        for (Int j = 0; j < size[1]; j ++) {
            y[j] = gmesh->y[j + offset[1]];
            dy[j] = gmesh->dy[j + offset[1]];
        }

        z = new Real[size[2]];
        dz = new Real[size[2]];
        for (Int k = 0; k < size[2]; k ++) {
            z[k] = gmesh->z[k + offset[2]];
            dz[k] = gmesh->dz[k + offset[2]];
        }

#pragma acc enter data \
copyin(x[:size[0]], y[:size[1]], z[:size[2]]) \
copyin(dx[:size[0]], dy[:size[1]], dz[:size[2]]) 
    }

    void finalize(Int size[3]) {
        delete[] x;
        delete[] y;
        delete[] z;
        delete[] dx;
        delete[] dy;
        delete[] dz;

#pragma acc exit data \
delete(x[:size[0]], y[:size[1]], z[:size[2]]) \
delete(dx[:size[0]], dy[:size[1]], dz[:size[2]])
    }
};

struct Cfd {
    Real (*U)[3], (*Uold)[3];
    Real (*JU)[3];
    Real *p, *nut, *div;
    Real Uin[3];
    Real Re, Cs;
    Real avg_div, max_cfl;

    void initialize(Real Uin[3], Real Re, Real Cs, Int size[3]) {
        this->Uin[0] = Uin[0];
        this->Uin[1] = Uin[1];
        this->Uin[2] = Uin[2];
        this->Re = Re;
        this->Cs = Cs;

        Int len = size[0]*size[1]*size[2];
        U = new Real[len][3];
        Uold = new Real[len][3];
        JU = new Real[len][3];
        p = new Real[len];
        nut = new Real[len];
        div = new Real[len];

#pragma acc enter data \
create(U[:len], Uold[:len], JU[:len], p[:len], nut[:len], div[:len])

        // printf("CFD INFO\n");
        // printf("\tRe = %lf\n", this->Re);
        // printf("\tCs = %lf\n", this->Cs);
        // printf("\tUin = (%lf %lf %lf)\n", this->Uin[0], this->Uin[1], this->Uin[2]);
    }

    void finalize(Int size[3]) {
        delete[] U;
        delete[] Uold;
        delete[] JU;
        delete[] p;
        delete[] nut;
        delete[] div;

        Int len = size[0]*size[1]*size[2];
#pragma acc exit data \
delete(U[:len], Uold[:len], JU[:len], p[:len], nut[:len], div[:len])
    }
};

struct Eq {
    Real (*A)[7];
    Real *b, *r;
    Int it, max_it;
    Real err, tol;
    Real max_diag;
    Real relax_rate = 1.2;

    void initialize(Int max_it, Real tol, Int size[3]) {
        this->max_it = max_it;
        this->tol = tol;

        Int len = size[0]*size[1]*size[2];
        A = new Real[len][7];
        b = new Real[len];
        r = new Real[len];

#pragma acc enter data \
create(A[:len], b[:len], r[:len])

        // printf("EQ INFO\n");
        // printf("\tmax iteration = %ld\n", this->max_it);
        // printf("\ttolerance = %lf\n", this->tol);
    }

    void finalize(Int size[3]) {
        delete[] A;
        delete[] b;
        delete[] r;

        Int len = size[0]*size[1]*size[2];
#pragma acc exit data \
delete(A[:len], b[:len], r[:len])
    }
};

struct Solver {
    Int gsize[3];
    Int size[3];
    Int offset[3];
    Int gc = 2;

    MpiInfo mpi;
    Runtime rt;
    Mesh gmesh, mesh;
    Cfd cfd;
    Eq eq;

    json setup_json;
    json snapshot_json = {};
    string output_prefix;

    void initialize(string path) {
        MPI_Comm_size(MPI_COMM_WORLD, &mpi.size);
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi.rank);

        int gpu_count = acc_get_num_devices(acc_device_nvidia);
        int gpu_id = mpi.rank%gpu_count;
        acc_set_device_num(gpu_id, acc_device_nvidia);

        ifstream setup_file(path);
        setup_json = json::parse(setup_file);

        auto &rt_json = setup_json["runtime"];
        Real dt = rt_json["dt"];
        Real total_time = rt_json["time"];
        rt.initialize(total_time/dt, dt);

        auto &output_json = setup_json["output"];
        output_prefix = output_json["prefix"];

        auto &mesh_json = setup_json["mesh"];
        string mesh_path = mesh_json["path"];
        gmesh.initialize_from_path(mesh_path, gsize, gc, &mpi);
        mesh.initialize_from_global_mesh(&gmesh, gsize, size, offset, gc, &mpi);

        // printf("%d mesh OK\n", mpi.rank);

        auto &inflow_json = setup_json["inflow"];
        auto &cfd_json = setup_json["cfd"];
        Real Uin[3];
        Uin[0] = inflow_json["value"][0];
        Uin[1] = inflow_json["value"][1];
        Uin[2] = inflow_json["value"][2];
        Real Re = cfd_json["Re"];
        Real Cs = cfd_json["Cs"];
        cfd.initialize(Uin, Re, Cs, size);

        // printf("%d cfd OK\n", mpi.rank);

        auto &eq_json = setup_json["eq"];
        Real tol = eq_json["tolerance"];
        Real max_it = eq_json["max_iteration"];
        eq.initialize(max_it, tol, size);

        // printf("%d eq OK\n", mpi.rank);

        eq.max_diag = build_A(
            eq.A,
            mesh.x, mesh.y, mesh.z,
            mesh.dx, mesh.dy, mesh.dz,
            size, gc,
            &mpi
        );

        // printf("%d eq A OK\n", mpi.rank);

        Int len = size[0]*size[1]*size[2];
        fill_array(cfd.U, cfd.Uin, len);
        fill_array(cfd.Uold, cfd.Uin, len);
        fill_array(cfd.p, 0., len);

        apply_Ubc(
            cfd.U, cfd.Uold, cfd.Uin,
            mesh.dx, mesh.dy, mesh.dz,
            rt.dt,
            size, gc,
            &mpi
        );

        // printf("%d Ubc OK\n", mpi.rank);

        interpolate_JU(
            cfd.U, cfd.JU,
            mesh.dx, mesh.dy, mesh.dz,
            size, gc,
            &mpi
        );

        // printf("%d JU OK\n", mpi.rank);

        apply_JUbc(
            cfd.JU, cfd.Uin,
            mesh.dy, mesh.dz,
            size, gc,
            &mpi
        );

        // printf("%d JUbc OK\n", mpi.rank);

        calc_nut(
            cfd.U, cfd.nut,
            mesh.x, mesh.y, mesh.z,
            mesh.dx, mesh.dy, mesh.dz,
            cfd.Cs,
            size, gc,
            &mpi
        );

        // printf("%d nut OK\n", mpi.rank);

        calc_divergence(
            cfd.JU, cfd.div,
            mesh.dx, mesh.dy, mesh.dz,
            size, gc,
            &mpi
        );

        // printf("%d div OK\n", mpi.rank);

        Int effective_count = (gsize[0] - 2*gc)*(gsize[1] - 2*gc)*(gsize[2] - 2*gc);
        cfd.avg_div = calc_l2_norm(cfd.div, size, gc, &mpi)/sqrt(effective_count);
        cfd.max_cfl = calc_max_cfl(cfd.U, mesh.dx, mesh.dy, mesh.dz, rt.dt, size, gc, &mpi);

        // printf("%d ||div|| OK\n", mpi.rank);
        if (mpi.rank == 0) {
            printf("SETUP INFO\n");
            printf("\tpath %s\n", path.c_str());

            printf("DEVICE INFO\n");
            printf("\tnumber of GPUs = %d\n", gpu_count);

            printf("MESH INFO\n");
            printf("\tpath = %s\n", mesh_path.c_str());
            printf("\tglobal size = (%ld %ld %ld)\n", gsize[0], gsize[1], gsize[2]);
            printf("\tguide cell = %ld\n", gc);

            printf("RUNTIME INFO\n");
            printf("\tdt = %lf\n", rt.dt);
            printf("\tmax step = %ld\n", rt.max_step);

            printf("OUTPUT INFO\n");
            printf("\tprefix = %s\n", output_prefix.c_str());

            printf("CFD INFO\n");
            printf("\tRe = %lf\n", cfd.Re);
            printf("\tCs = %lf\n", cfd.Cs);
            printf("\tinitial div(U) = %e\n", cfd.avg_div);
            printf("\tinitial max cfd = %e\n", cfd.max_cfl);

            printf("EQ INFO\n");
            printf("\tmax iteration = %ld\n", eq.max_it);
            printf("\ttolerance = %lf\n", eq.tol);
            printf("\tmax A diag = %lf\n", eq.max_diag);

            write_mesh(
                output_prefix + "_mesh.txt",
                gmesh.x, gmesh.y, gmesh.z,
                gmesh.dx, gmesh.dy, gmesh.dz,
                gsize, gc
            );

            ofstream json_output(output_prefix + ".json");
            json_output << setw(2) << setup_json;
            json_output.close();

            json part_json;
            part_json["size"] = {gsize[0], gsize[1], gsize[2]};
            part_json["gc"] = gc;
            part_json["partition"] = {};
            for (Int rank = 0; rank < mpi.size; rank ++) {
                json rank_json;
                Int rank_size[3], rank_offset[3];
                MpiInfo rank_info;
                rank_info.size = mpi.size;
                rank_info.rank = rank;
                calc_partition(gsize, rank_size, rank_offset, gc, &rank_info);
                rank_json["size"] = {rank_size[0], rank_size[1], rank_size[2]};
                rank_json["offset"] = {rank_offset[0], rank_offset[1], rank_offset[2]};
                rank_json["rank"] = rank;
                part_json["partition"].push_back(rank_json);
            }
            ofstream part_output(output_prefix + "_partition.json");
            part_output << setw(2) << part_json;
            part_output.close();
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        for (Int rank = 0; rank < mpi.size; rank ++) {
            if (mpi.rank == rank) {
                printf("PROC INFO %d/%d\n", mpi.rank, mpi.size);
                printf("\tsize = (%ld %ld %ld)\n", size[0], size[1], size[2]);
                printf("\toffset = (%ld %ld %ld)\n", offset[0], offset[1], offset[2]);
                printf("\tGPU id = %d\n", gpu_id);
            }
            fflush(stdout);
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    void finalize() {
        if (mpi.rank == 0) {
            ofstream snapshot_output(output_prefix + "_snapshot.json");
            snapshot_output << setw(2) << snapshot_json;
            snapshot_output.close();
        }
        gmesh.finalize(gsize);
        mesh.finalize(size);
        cfd.finalize(size);
        eq.finalize(size);
    }

    void main_loop_once() {
        Int len = size[0]*size[1]*size[2];

        cpy_array(cfd.Uold, cfd.U, len);

        // printf("1\n");

        calc_intermediate_U(
            cfd.U, cfd.Uold, cfd.JU, cfd.nut,
            mesh.x, mesh.y, mesh.z,
            mesh.dx, mesh.dy, mesh.dz,
            cfd.Re, rt.dt,
            size, gc,
            &mpi
        );

        interpolate_JU(
            cfd.U, cfd.JU,
            mesh.dx, mesh.dy, mesh.dz,
            size, gc,
            &mpi
        );

        apply_JUbc(
            cfd.JU, cfd.Uin,
            mesh.dy, mesh.dz,
            size, gc,
            &mpi
        );

        // printf("2\n");

        calc_poisson_rhs(
            cfd.JU, eq.b,
            mesh.dx, mesh.dy, mesh.dz,
            rt.dt, eq.max_diag,
            size, gc,
            &mpi
        );

        // printf("3\n");

        run_sor(
            eq.A, cfd.p, eq.b, eq.r,
            eq.relax_rate, eq.it, eq.max_it, eq.err, eq.tol,
            gsize, size, offset, gc,
            &mpi
        );

        // printf("4\n");

        apply_pbc(
            cfd.Uold, cfd.p, cfd.nut,
            mesh.z, mesh.dx, mesh.dy, mesh.dz,
            cfd.Re,
            size, gc,
            &mpi
        );

        // printf("5\n");

        project_p(
            cfd.U, cfd.JU, cfd.p,
            mesh.x, mesh.y, mesh.z,
            mesh.dx, mesh.dy, mesh.dz,
            rt.dt,
            size, gc,
            &mpi
        );

        // printf("6\n");

        apply_Ubc(
            cfd.U, cfd.Uold, cfd.Uin,
            mesh.dx, mesh.dy, mesh.dz,
            rt.dt,
            size, gc,
            &mpi
        );

        apply_JUbc(
            cfd.JU, cfd.Uin,
            mesh.dy, mesh.dz,
            size, gc,
            &mpi
        );

        // printf("7\n");

        calc_nut(
            cfd.U, cfd.nut,
            mesh.x, mesh.y, mesh.z,
            mesh.dx, mesh.dy, mesh.dz,
            cfd.Cs,
            size, gc,
            &mpi
        );

        calc_divergence(
            cfd.JU, cfd.div,
            mesh.dx, mesh.dy, mesh.dz,
            size, gc,
            &mpi
        );

        // printf("8\n");

        Int effective_count = (gsize[0] - 2*gc)*(gsize[1] - 2*gc)*(gsize[2] - 2*gc);
        cfd.avg_div = calc_l2_norm(cfd.div, size, gc, &mpi)/sqrt(effective_count);

        cfd.max_cfl = calc_max_cfl(cfd.U, mesh.dx, mesh.dy, mesh.dz, rt.dt, size, gc, &mpi);

        // printf("9\n");

        rt.step ++;

        if (mpi.rank == 0) {
            printf("%ld %e %ld %e %e %e\n", rt.step, rt.get_time(), eq.it, eq.err, cfd.avg_div, cfd.max_cfl);
        }
    }
};

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    Solver solver;
    string setup_path(argv[1]);
    solver.initialize(setup_path);
    Int *size = solver.size;
    Int len = size[0]*size[1]*size[2];

    Header header;
    header.size[0] = solver.size[0];
    header.size[1] = solver.size[1];
    header.size[2] = solver.size[2];
    header.gc = solver.gc;
    header.var_count = 3;
    header.var_dim = {3, 1, 1};
    header.var_name = {"U", "p", "div"};
    Real *var[] = {solver.cfd.U[0], solver.cfd.p, solver.cfd.div};

// #pragma acc update \
// host(solver.cfd.U[:len], solver.cfd.p[:len], solver.cfd.div[:len])
//     // write_csv(
//     //     "data/0.csv",
//     //     var, var_count, var_dim, var_name,
//     //     solver.mesh.x, solver.mesh.y, solver.mesh.z,
//     //     solver.size, solver.gc
//     // );

    for (; solver.rt.step < solver.rt.max_step;) {
        solver.main_loop_once();
    }    

#pragma acc update \
host(solver.cfd.U[:len], solver.cfd.p[:len], solver.cfd.div[:len])
    string filename = make_rank_binary_filename(solver.output_prefix, solver.mpi.rank, solver.rt.step);
    write_binary(
        filename, &header,
        var, solver.mesh.x, solver.mesh.y, solver.mesh.z
    );
    json slice_json = {{"step", solver.rt.step}, {"time", solver.rt.get_time()}};
    solver.snapshot_json.push_back(slice_json);

    solver.finalize();
    MPI_Finalize();
}