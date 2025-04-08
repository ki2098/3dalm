#include <cmath>
#include "mv.h"
#include "util.h"

void cpy_array(
    double dst[],
    double src[],
    int sz
) {
    #pragma acc parallel loop independent \
    present(dst[:sz], src[:sz]) \
    firstprivate(sz)
    for (int i = 0; i < sz; i ++) {
        dst[i] = src[i];
    }
}

void clear_array(
    double dst[],
    int sz
) {
    #pragma acc parallel loop independent \
    present(dst[:sz]) \
    firstprivate(sz)
    for (int i = 0; i < sz; i ++) {
        dst[i] = 0;
    }
}

double calc_norm(
    double x[],
    int sz[3],
    int gc,
    mpi_info *mpi
) {
    int cnt = sz[0]*sz[1]*sz[2];

    double total = 0;

    #pragma acc parallel loop independent reduction(+:total) collapse(3) \
    present(x[:cnt]) \
    firstprivate(sz[:3], gc)
    for (int i = gc; i < sz[0] - gc; i ++) {
        for (int j = gc; j < sz[1] - gc; j ++) {
            for (int k = gc; k < sz[2] - gc; k ++) {
                total += square(x[getid(i, j, k, sz)]);
            }
        }
    }

    return sqrt(total);
}

double calc_dot_product(
    double a[],
    double b[],
    int sz[3],
    int gc,
    mpi_info *mpi
) {
    int cnt = sz[0]*sz[1]*sz[2];

    double total = 0;

    #pragma acc parallel loop independent reduction(+:total) collapse(3) \
    present(a[:cnt], b[:cnt]) \
    firstprivate(sz[:3], gc)
    for (int i = gc; i < sz[0] - gc; i ++) {
        for (int j = gc; j < sz[1] - gc; j ++) {
            for (int k = gc; k < sz[2] - gc; k ++) {
                int id = getid(i, j, k, sz);
                total += a[id]*b[id];
            }
        }
    }
    return total;
}

double calc_Ax(
    double A[][7],
    double x[],
    double y[],
    int sz[3],
    int gc,
    mpi_info *mpi
) {
    int cnt = sz[0]*sz[1]*sz[2];

    #pragma acc parallel loop independent collapse(3) \
    present(A[:cnt], x[:cnt], y[:cnt]) \
    firstprivate(sz[:3], gc)
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
                double ac = A[idc][0];
                double ae = A[idc][1];
                double aw = A[idc][2];
                double an = A[idc][3];
                double as = A[idc][4];
                double at = A[idc][5];
                double ab = A[idc][6];
                double xc = x[idc];
                double xe = x[ide];
                double xw = x[idw];
                double xn = x[idn];
                double xs = x[ids];
                double xt = x[idt];
                double xb = x[idb];
                y[idc] = ac*xc + ae*xe + aw*xw + an*xn + as*xs + at*xt + ab*xb;
            }
        }
    }
}

double calc_residual(
    double A[][7],
    double x[],
    double b[],
    double r[],
    int sz[3],
    int gc,
    mpi_info *mpi
) {
    int cnt = sz[0]*sz[1]*sz[2];

    #pragma acc parallel loop independent collapse(3) \
    present(A[:cnt], x[:cnt], b[:cnt], r[:cnt]) \
    firstprivate(sz[:3], gc)
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
                double ac = A[idc][0];
                double ae = A[idc][1];
                double aw = A[idc][2];
                double an = A[idc][3];
                double as = A[idc][4];
                double at = A[idc][5];
                double ab = A[idc][6];
                double xc = x[idc];
                double xe = x[ide];
                double xw = x[idw];
                double xn = x[idn];
                double xs = x[ids];
                double xt = x[idt];
                double xb = x[idb];
                r[idc] = b[idc] - ac*xc + ae*xe + aw*xw + an*xn + as*xs + at*xt + ab*xb;
            }
        }
    }
}