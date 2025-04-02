#include "pbicgstab.h"

void sweep(
    double A[][7],
    double x[],
    double x_tmp[],
    double b[],
    int sz[3],
    int gc
) {

}

void run_preconditioner(
    double A[][7],
    double x[],
    double x_tmp[],
    double b[],
    int max_iteration,
    int sz[3],
    int gc,
    mpi_info mpi
) {
    for (int it = 0; it < max_iteration; it ++) {
        for (int i = 0; i < sz[0]*sz[1]*sz[2]; i ++) {
            x_tmp[i] = x[i];
        }

        sweep(A, x, x_tmp, b, sz, gc);
    }
}