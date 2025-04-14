#pragma once

#include "mpi_info.h"

void cpy_array(
    double dst[],
    double src[],
    int sz
);

template<int N>
void cpy_array(
    double dst[][N],
    double src[][N],
    int sz
) {
    #pragma acc data present(dst[:sz], src[:sz])
    #pragma acc parallel firstprivate(sz)
    #pragma acc loop independent
    for (int i = 0; i < sz; i ++) {
        for (int m = 0; m < N; m ++) {
            dst[i][m] = src[i][m];
        }
    }
}

void clear_array(
    double dst[],
    int sz
);

template<int N>
void clear_array(
    double dst[][N],
    int sz
) {
    #pragma acc data present(dst[:sz])
    #pragma acc parallel firstprivate(sz)
    #pragma acc loop independent
    for (int i = 0; i < sz; i ++) {
        for (int m = 0; m < N; m ++) {
            dst[i][m] = 0;
        }
    }
}

void fill_array(
    double dst[],
    double value,
    int sz
);

template<int N>
void fill_array(
    double dst[][N],
    double value[N],
    int sz
) {
    #pragma acc data present(dst[:sz])
    #pragma acc parallel firstprivate(value[:N], sz)
    #pragma acc loop independent
    for (int i = 0; i < sz; i ++) {
        for (int m = 0; m < N; m ++) {
            dst[i][m] = value[m];
        }
    }
}

double calc_norm(
    double x[],
    int sz[3],
    int gc,
    mpi_info *mpi
);

double calc_dot_product(
    double a[],
    double b[],
    int sz[3],
    int gc,
    mpi_info *mpi
);

void calc_Ax(
    double A[][7],
    double x[],
    double y[],
    int sz[3],
    int gc,
    mpi_info *mpi
);

void calc_residual(
    double A[][7],
    double x[],
    double b[],
    double r[],
    int sz[3],
    int gc,
    mpi_info *mpi
);