#pragma once

#include "util.h"

static void apply_Ubc(
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

static void apply_JUbc(
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

static void apply_pbc(
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