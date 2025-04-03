#include <cmath>
#include <cfloat>
#include "pbicgstab.h"
#include "mv.h"
#include "util.h"

void sweep(
    double A[][7],
    double x[],
    double x_tmp[],
    double b[],
    int sz[3],
    int gc
) {
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
                double xc = x_tmp[idc];
                double xe = x_tmp[ide];
                double xw = x_tmp[idw];
                double xn = x_tmp[idn];
                double xs = x_tmp[ids];
                double xt = x_tmp[idt];
                double xb = x_tmp[idb];
                x[idc] = xc + (b[idc] - (ac*xc + ae*xe + aw*xw + an*xn + as*xs + at*xt + ab*xb))/ac;
            }
        }
    }
}

void run_preconditioner(
    double A[][7],
    double x[],
    double x_tmp[],
    double b[],
    int max_iteration,
    int sz[3],
    int gc,
    mpi_info *mpi
) {
    for (int it = 0; it < max_iteration; it ++) {
        cpy_array(x_tmp, x, sz[0]*sz[1]*sz[2]);
        sweep(A, x, x_tmp, b, sz, gc);
    }
}

void run_pbicgstab(
    double A[][7],
    double x[],
    double b[],
    double r[],
    double r0[],
    double p[],
    double q[],
    double s[],
    double p_hat[],
    double s_hat[],
    double t[],
    double tmp[],
    double &err,
    double tolerance,
    int &it,
    int max_iteration,
    int sz[3],
    int gc,
    mpi_info *mpi
) {
    int cnt = sz[0]*sz[1]*sz[2];
    int effective_cnt = (sz[0] - 2*gc)*(sz[1] - 2*gc)*(sz[2] - 2*gc);
    double rho;
    double rho_old = 1;
    double alpha = 0;
    double omega = 1;
    double beta;

    calc_residual(A, x, b, r, sz, gc, mpi);
    err = calc_norm(r, sz, gc, mpi)/sqrt(effective_cnt);

    cpy_array(r0, r, cnt);

    it = 0;
    do {
        rho = calc_dot_product(r, r0, sz, gc, mpi);
        if (fabs(rho) < FLT_MIN) {
            err = rho;
            break;
        }

        if (it == 0) {
            cpy_array(p, r, cnt);
        } else {
            beta = (rho/rho_old)*(alpha/omega);
            for (int i = 0; i < cnt; i ++) {
                p[i] = r[i] + beta*(p[i] - omega*q[i]);
            }
        }

        clear_array(p_hat, cnt);
        run_preconditioner(A, p_hat, tmp, p, 3, sz, gc, mpi);

        calc_Ax(A, p_hat, q, sz, gc, mpi);

        alpha = rho / calc_dot_product(r0, q, sz, gc, mpi);

        for (int i = 0; i < cnt; i ++) {
            s[i] = r[i] - alpha*q[i];
        }

        clear_array(s_hat, cnt);
        run_preconditioner(A, s_hat, tmp, s, 3, sz, gc, mpi);

        calc_Ax(A, s_hat, t, sz, gc, mpi);

        omega = calc_dot_product(t, s, sz, gc, mpi)/calc_dot_product(t, t, sz, gc, mpi);

        for (int i = 0; i < cnt; i ++) {
            x[i] = x[i] + alpha*p_hat[i] + omega*s_hat[i];
        }

        for (int i = 0; i < cnt; i ++) {
            r[i] = s[i] - omega*t[i];
        }

        rho_old = rho;

        err = calc_norm(r, sz, gc, mpi)/sqrt(effective_cnt);
        it ++;
    } while (it < max_iteration && err > tolerance);
}