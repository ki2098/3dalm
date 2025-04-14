#include <cmath>
#include <cfloat>
#include <cstdio>
#include <iostream>
#include "pbicgstab.h"
#include "mv.h"
#include "util.h"

void jacobi_sweep(
    double A[][7],
    double x[],
    double xtmp[],
    double b[],
    int sz[3],
    int gc
) {
    int cnt = sz[0]*sz[1]*sz[2];

    #pragma acc data present(A[:cnt], x[:cnt], xtmp[:cnt], b[:cnt])
    #pragma acc parallel firstprivate(sz[:3], gc)
    #pragma acc loop independent collapse(3)
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
                double xc = xtmp[idc];
                double xe = xtmp[ide];
                double xw = xtmp[idw];
                double xn = xtmp[idn];
                double xs = xtmp[ids];
                double xt = xtmp[idt];
                double xb = xtmp[idb];
                x[idc] = xc + (b[idc] - (ac*xc + ae*xe + aw*xw + an*xn + as*xs + at*xt + ab*xb))/ac;
                // if (ac == 0) {
                //     printf("id %d %d %d %d\n", i, j, k, idc);
                //     // printf("A %e %e %e %e %e %e %e\n", ac, ae, aw, an, as, at, ab);
                //     // printf("x %e %e %e %e %e %e %e\n", xc, xe, xw, xn, xs, xt, xb);
                //     // printf("b %e\n", b[idc]);
                //     // printf("new x %e\n", x[idc]);
                // }
            }
        }
    }
}

void sor_sweep(
    double A[][7],
    double x[],
    double b[],
    double omega,
    int color,
    int sz[3],
    int gc
) {
    int cnt = sz[0]*sz[1]*sz[2];
    #pragma acc data present(A[:cnt], x[:cnt], b[:cnt])
    #pragma acc parallel firstprivate(sz[:3], gc, omega, color)
    #pragma acc loop independent collapse(3)
    for (int i = gc; i < sz[0] - gc; i ++) {
    for (int j = gc; j < sz[1] - gc; j ++) {
    for (int k = gc; k < sz[2] - gc; k ++) {
        if ((i + j + k)%2 == color) {
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
            x[idc] = xc + omega*(b[idc] - (ac*xc + ae*xe + aw*xw + an*xn + as*xs + at*xt + ab*xb))/ac;
        }
    }}}
}

void run_sor(
    double A[][7],
    double x[],
    double b[],
    double r[],
    double omega,
    double &err,
    double tolerance,
    int &it,
    int max_iteration,
    int sz[3],
    int gc,
    mpi_info *mpi
) {
    it = 0;
    int effective_cnt = (sz[0] - 2*gc)*(sz[1] - 2*gc)*(sz[2] - 2*gc);
    do {
        sor_sweep(A, x, b, omega, 0, sz, gc);
        sor_sweep(A, x, b, omega, 1, sz, gc);
        calc_residual(A, x, b, r, sz, gc, mpi);
        err = calc_norm(r, sz, gc, mpi)/sqrt(effective_cnt);
        it ++;
    } while (it < max_iteration && err > tolerance);
}

void run_preconditioner(
    double A[][7],
    double x[],
    double xtmp[],
    double b[],
    int max_iteration,
    int sz[3],
    int gc,
    mpi_info *mpi
) {
    for (int it = 0; it < max_iteration; it ++) {
        cpy_array(xtmp, x, sz[0]*sz[1]*sz[2]);
        jacobi_sweep(A, x, xtmp, b, sz, gc);
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
    double phat[],
    double shat[],
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
    // printf("initial err %e\n", err);

    cpy_array(r0, r, cnt);

    it = 0;
    do {
        rho = calc_dot_product(r, r0, sz, gc, mpi);
        if (fabs(rho) < FLT_MIN) {
            err = rho;
            break;
        }
        // printf("rho %e\n", rho);

        if (it == 0) {
            cpy_array(p, r, cnt);
        } else {
            beta = (rho/rho_old)*(alpha/omega);

            #pragma acc data present(p[:cnt], q[:cnt], r[:cnt])
            #pragma acc parallel firstprivate(beta, omega, cnt)
            #pragma acc loop independent
            for (int i = 0; i < cnt; i ++) {
                p[i] = r[i] + beta*(p[i] - omega*q[i]);
            }
        }

        clear_array(phat, cnt);
        run_preconditioner(A, phat, tmp, p, 2, sz, gc, mpi);
        // printf("p^ %e\n", calc_norm(phat, sz, gc, mpi));

        calc_Ax(A, phat, q, sz, gc, mpi);

        alpha = rho / calc_dot_product(r0, q, sz, gc, mpi);

        #pragma acc data present(s[:cnt], r[:cnt], q[:cnt])
        #pragma acc parallel firstprivate(alpha, cnt)
        #pragma acc loop independent
        for (int i = 0; i < cnt; i ++) {
            s[i] = r[i] - alpha*q[i];
        }

        clear_array(shat, cnt);
        run_preconditioner(A, shat, tmp, s, 2, sz, gc, mpi);

        calc_Ax(A, shat, t, sz, gc, mpi);

        omega = calc_dot_product(t, s, sz, gc, mpi)/calc_dot_product(t, t, sz, gc, mpi);
        // printf("omega %e\n", omega);

        #pragma acc data present(x[:cnt], phat[:cnt], shat[:cnt])
        #pragma acc parallel firstprivate(alpha, omega, cnt)
        #pragma acc loop independent
        for (int i = 0; i < cnt; i ++) {
            x[i] = x[i] + alpha*phat[i] + omega*shat[i];
        }

        #pragma acc data present(r[:cnt], s[:cnt], t[:cnt])
        #pragma acc parallel firstprivate(omega, cnt)
        #pragma acc loop independent
        for (int i = 0; i < cnt; i ++) {
            r[i] = s[i] - omega*t[i];
        }

        rho_old = rho;

        err = calc_norm(r, sz, gc, mpi)/sqrt(effective_cnt);
        // printf("err %e\n", err);
        it ++;
    } while (it < max_iteration && err > tolerance);
}

double build_coefficient_matrix(
    double A[][7],
    double dx[],
    double dy[],
    double dz[],
    int sz[3],
    int gc,
    mpi_info *mpi
) {
    int cnt = sz[0]*sz[1]*sz[2];

    double max_diag = 0;

    #pragma acc data present(A[:cnt], dx[:sz[0]], dy[:sz[1]], dz[:sz[2]])
    #pragma acc parallel firstprivate(sz[:3], gc)
    #pragma acc loop independent collapse(3) reduction(max:max_diag)
    for (int i = gc; i < sz[0] - gc; i ++) {
        for (int j = gc; j < sz[1] - gc; j ++) {
            for (int k = gc; k < sz[2] - gc; k ++) {
                double xxc = 1/dx[i];
                double xxe = 1/dx[i + 1];
                double xxw = 1/dx[i - 1];
                double yyc = 1/dy[j];
                double yyn = 1/dy[j + 1];
                double yys = 1/dy[j - 1];
                double zzc = 1/dz[k];
                double zzt = 1/dz[k + 1];
                double zzb = 1/dz[k - 1];
                double ac = 0;
                double ae = 0;
                double aw = 0;
                double an = 0;
                double as = 0;
                double at = 0;
                double ab = 0;
                if (i < sz[0] - gc) {
                    double coefficient = xxc*(xxc + 0.25*(xxe - xxw));
                    ae  = coefficient;
                    ac -= coefficient;
                }
                if (i > gc) {
                    double coefficient = xxc*(xxc - 0.25*(xxe - xxw));
                    aw  = coefficient;
                    ac -= coefficient;
                }
                if (j < sz[1] - gc - 1) {
                    double coefficient = yyc*(yyc + 0.25*(yyn - yys));
                    an  = coefficient;
                    ac -= coefficient;
                }
                if (j > gc) {
                    double coefficient = yyc*(yyc - 0.25*(yyn - yys));
                    as  = coefficient;
                    ac -= coefficient;
                }
                if (k < sz[2] - gc - 1) {
                    double coefficient = zzc*(zzc + 0.25*(zzt - zzb));
                    at  = coefficient;
                    ac -= coefficient;
                }
                if (k > gc) {
                    double coefficient = zzc*(zzc - 0.25*(zzt - zzb));
                    ab  = coefficient;
                    ac -= coefficient;
                }
                int id = getid(i, j, k, sz);
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
                // if (i >= 600) {
                //     printf("%d %d %d %d %e %e %e %e %e %e %e\n", i, j, k, sz[0] - gc, ac, ae, aw, an, as, at, ab);
                // }
            }
        }
    }

    // int ix = sz[0], jx = sz[1], kx = sz[2];
    // #pragma acc data present(A[:cnt])
    // #pragma acc parallel copyin(ix, jx, kx, gc)
    // #pragma acc loop independent collapse(3)
    // for (int i = 600; i < ix - gc; i ++) {
    //     for (int j = gc; j < jx - gc; j ++) {
    //         for (int k = gc; k < kx - gc; k ++) {
    //             int id = i*jx*kx + j*kx + k;
    //             printf("ID %d %d %d %d\n", i, j, k, id);
    //             // printf("%d %d %d %d %d %d %d %e %e %e %e %e %e %e\n", i, j, k, sz[0], sz[1], sz[2], gc, A[id][0], A[id][1], A[id][2], A[id][3], A[id][4], A[id][5], A[id][6]);
    //         }
    //     }
    // }

    #pragma acc data present(A[:cnt])
    #pragma acc parallel firstprivate(max_diag, cnt)
    #pragma acc loop independent collapse(2)
    for (int i = 0; i < cnt; i ++) {
        for (int m = 0; m < 7; m ++) {
            A[i][m] /= max_diag;
        }
    }

    return max_diag;
}
