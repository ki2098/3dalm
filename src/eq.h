#pragma once

#include "mv.h"

static void sweep_sor(
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

static void run_sor(
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

static Real build_A(
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