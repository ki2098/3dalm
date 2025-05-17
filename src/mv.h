#pragma once

#include <cmath>
#include <mpi.h>
#include "util.h"
#include "mpi_type.h"

static Real calc_inner_product(
    Real a[], Real b[],
    Int size[3], Int gc,
    MpiInfo *mpi
) {
    Real total = 0;
    Int len = size[0]*size[1]*size[2];
#pragma acc kernels loop independent collapse(3) \
present(a[:len], b[:len]) \
copyin(size[:3]) \
reduction(+:total)
    for (Int i = gc; i < size[0] - gc; i ++) {
    for (Int j = gc; j < size[1] - gc; j ++) {
    for (Int k = gc; k < size[2] - gc; k ++) {
        Int id = index(i, j, k, size);
        total += a[id]*b[id];
    }}}

    if (mpi->size > 1) {
        MPI_Allreduce(MPI_IN_PLACE, &total, 1, get_mpi_datatype<Real>(), MPI_SUM, MPI_COMM_WORLD);
    }

    return total;
}

static Real calc_l2_norm(
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

static void calc_residual(
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

static void calc_Ax(
    Real A[][7], Real x[], Real y[],
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
present(A[:len], x[:len], y[:len]) \
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

        y[idc] = ac*xc + ae*xe + aw*xw + an*xn + as*xs + at*xt + ab*xb;
    }}}
}

static void calc_ax_plus_by(
    Real a, Real x[], Real b, Real y[], Real z[],
    Int len
) {
#pragma acc kernels loop independent \
present(x[:len], y[:len], z[:len])
    for (Int i = 0; i < len; i ++) {
        z[i] = a*x[i] + b*y[i];
    }
}

template<Int N>
void calc_ax_plus_by(
    Real a, Real x[][N], Real b, Real y[][N], Real z[][N],
    Int len
) {
#pragma acc kernels loop independent \
present(x[:len], y[:len], z[:len])
    for (Int i = 0; i < len; i ++) {
#pragma acc loop seq
        for (Int m = 0; m < N; m ++) {
            z[i][m] = a*x[i][m] + b*y[i][m];
        }
    }
}