#include <cmath>
#include "io.h"
#include "json.hpp"

using namespace std;
using json = nlohmann::json;

Real calc_convection_kk(Real stencil[13], Real U[3], Real dxyz[3]) {
    Real valc  = stencil[0];
    Real vale  = stencil[1];
    Real valee = stencil[2];
    Real valw  = stencil[3];
    Real valww = stencil[4];
    Real valn  = stencil[5];
    Real valnn = stencil[6];
    Real vals  = stencil[7];
    Real valss = stencil[8];
    Real valt  = stencil[9];
    Real valtt = stencil[10];
    Real valb  = stencil[11];
    Real valbb = stencil[12];
    Real u = U[0];
    Real v = U[1];
    Real w = U[2];
    Real dx = dxyz[0];
    Real dy = dxyz[1];
    Real dz = dxyz[2];

    Real convection =
        u*(- valee + 8*vale - 8*valw + valww)/(12*dx)
    + 0.25*fabs(u)*(valee - 4*vale + 6*valc - 4*valw + valww)/(dx)
    +   v*(- valnn + 8*valn - 8*vals + valss)/(12*dy)
    + 0.25*fabs(v)*(valnn - 4*valn + 6*valc - 4*vals + valss)/(dy)
    +   w*(- valtt + 8*valt - 8*valb + valbb)/(12*dz)
    + 0.25*fabs(w)*(valtt - 4*valt + 6*valc - 4*valb + valbb)/(dz);

    return convection;
}

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
    Real U[][3], Real Uold[][3], Real nut[],
    Real x[], Real y[], Real z[],
    Real dx[], Real dy[], Real dz[],
    Real Re, Real dt,
    Int size[3], Int gc,
    MpiInfo *mpi
) {
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
            Real convection = calc_convection_kk(convection_stencil, Uold[idc], dxyz);

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

void calc_poisson_rhs(
    Real U[][3], Real rhs[],
    Real x[], Real y[], Real z[],
    Real dx[], Real dy[], Real dz[],
    Real dt, Real scale,
    Int size[3], Int gc,
    MpiInfo *mpi
) {
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
        Real divergence = 
            (U[ide][0] - U[idw][0])/(x[i + 1] - x[i - 1])
        +   (U[idn][1] - U[ids][1])/(y[j + 1] - y[j - 1])
        +   (U[idt][2] - U[idb][2])/(z[k + 1] - z[k - 1]);
        rhs[idc] = divergence/(dt*scale);
    }}}
}

void project_p(
    Real U[][3], Real p[],
    Real x[], Real y[], Real z[],
    Real dt,
    Int size[3], Int gc,
    MpiInfo *mpi
) {
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
}

void calc_nut(
    Real U[][3], Real nut[],
    Real x[], Real y[], Real z[],
    Real dx[], Real dy[], Real dz[],
    Real Cs,
    Int size[3], Int gc,
    MpiInfo *mpi
) {
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

void calc_divergence(
    Real U[][3], Real div[],
    Real x[], Real y[], Real z[],
    Int size[3], Int gc,
    MpiInfo *mpi
) {
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
        Real divergence = 
            (U[ide][0] - U[idw][0])/(x[i + 1] - x[i - 1])
        +   (U[idn][1] - U[ids][1])/(y[j + 1] - y[j - 1])
        +   (U[idt][2] - U[idb][2])/(z[k + 1] - z[k - 1]);
        div[idc] = divergence;
    }}}
}

Real calc_l2_norm(
    Real v[],
    Int size[3], Int gc,
    MpiInfo *mpi
) {
    Real total = 0;
    for (Int i = gc; i < size[0] - gc; i ++) {
    for (Int j = gc; j < size[1] - gc; j ++) {
    for (Int k = gc; k < size[2] - gc; k ++) {
        total += square(v[index(i, j, k, size)]);
    }}}
    return total;
}

void calc_residual(
    Real A[][7], Real x[], Real b[], Real r[],
    Int size[3], Int gc,
    MpiInfo *mpi
) {
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
    Int size[3], Int gc,
    MpiInfo *mpi
) {
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

        Real relaxation = (b[idc] - (ac*xc + ae*xe + aw*xw + an*xn + as*xs + at*xt + ab*xb))/ac;

        x[idc] = xc + relax_rate*relaxation;
    }}}
}

void run_sor(
    Real A[][7], Real x[], Real b[], Real r[],
    Real relax_rate, Int &it, Int max_it, Real &err, Real tol,
    Int size[3], Int gc,
    MpiInfo *mpi
) {
    Int effective_cnt = (size[0] - 2*gc)*(size[1] - 2*gc)*(size[2] - 2*gc);
    it = 0;
    do {
        sweep_sor(A, x, b, relax_rate, 0, size, gc, mpi);
        sweep_sor(A, x, b, relax_rate, 1, size, gc, mpi);
        calc_residual(A, x, b, r, size, gc, mpi);
        err = calc_l2_norm(r, size, gc, mpi)/sqrt(effective_cnt);
        it ++;
    } while (it < max_it && err > tol);
}

Real construct_A(
    Real A[][7],
    Real x[], Real y[], Real z[],
    Real dx[], Real dy[], Real dz[],
    Int size[3], Int gc,
    MpiInfo *mpi
) {
    Real max_diag = 0;
    for (Int i = gc; i < size[0] - gc; i ++) {
    for (Int j = gc; j < size[1] - gc; j ++) {
    for (Int k = gc; k < size[2] - gc; k ++) {
        Int id = index(i, j, k, size);
        Real dxc = dx[i];
        Real dyc = dy[j];
        Real dzc = dz[k];
        Real dxec = x[i + 1] - x[i];
        Real dxcw = x[i] - x[i - 1];
        Real dync = y[j + 1] - y[j];
        Real dycs = y[j] - y[j - 1];
        Real dztc = z[k + 1] - z[k];
        Real dzcb = z[k] - z[k - 1];
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

    for (Int i = gc; i < size[0] - gc; i ++) {
    for (Int j = gc; j < size[1] - gc; j ++) {
    for (Int k = gc; k < size[2] - gc; k ++) {
        Int id = index(i, j, k, size);
        for (Int m = 0; m < 7; m ++) {
            A[id][m] /= max_diag;
        }
    }}}

    return max_diag;
}

void apply_Ubc(
    Real U[][3], Real Uold[][3], Real Uin[3],
    Real x[], Real y[], Real z[],
    Real dx[], Real dy[], Real dz[],
    Real dt,
    Int size[3], Int gc,
    MpiInfo *mpi
) {
    /** x- fixed value inflow */
    for (Int i = 0; i < gc; i ++) {
    for (Int j = gc; j < size[1] - gc; j ++) {
    for (Int k = gc; k < size[2] - gc; k ++) {
        Int id = index(i, j, k, size);
        U[id][0] = Uin[0];
        U[id][1] = Uin[1];
        U[id][2] = Uin[2];
    }}}

    /** x+ convective outflow */
    for (Int i = size[0] - gc; i < size[0]; i ++) {
    for (Int j = gc; j < size[1] - gc; j ++) {
    for (Int k = gc; k < size[2] - gc; k ++) {
        Int id0 = index(i    , j, k, size);
        Int id1 = index(i - 1, j, k, size);
        Int id2 = index(i - 2, j, k, size);
        Real h1 = x[i] - x[i - 1];
        Real h2 = x[i] - x[i - 2];
        Real uout = U[id0][0];
        for (Int m = 0; m < 3; m ++) {
            Real f0 = Uold[id0][m];
            Real f1 = Uold[id1][m];
            Real f2 = Uold[id2][m];
            Real grad = (f0*(h2*h2 - h1*h1) - f1*h2*h2 + f2*h1*h1)/(h1*h2*h2 - h2*h1*h1);
            U[id0][m] = f0 - uout*dt*grad;
        }
    }}}

    /** y- slip */
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
    for (Int i = gc; i < size[0] - gc; i ++) {
    for (Int j = gc; j < size[1] - gc; j ++) {
        Int ki  = gc;
        Int kii = gc - 1;
        Int ko  = gc + 1;
        Int koo = gc + 2;
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

void apply_pbc(
    Real p[],
    Real dx[], Real dy[], Real dz[],
    Int size[3], Int gc,
    MpiInfo *mpi
) {
    /** x- grad = 0 */
    for (Int j = gc; j < size[1] - gc; j ++) {
    for (Int k = gc; k < size[2] - gc; k ++) {
        Int ii = gc;
        Int io = gc - 1;
        p[index(io, j, k, size)] = p[index(ii, j, k, size)];
    }}

    /** x+ value = 0 */
    for (Int j = gc; j < size[1] - gc; j ++) {
    for (Int k = gc; k < size[2] - gc; k ++) {
        Int ii = size[0] - gc - 1;
        Int io = size[0] - gc;
        Int hi = 0.5*dx[ii];
        Int ho = 0.5*dx[io];
        Real pbc = 0;
        p[index(io, j, k, size)] = pbc - (p[index(ii, j, k, size)] - pbc)*(ho/hi);
    }}

    /** y- grad = 0 */
    for (Int i = gc; i < size[0] - gc; i ++) {
    for (Int k = gc; k < size[2] - gc; k ++) {
        Int ji = gc;
        Int jo = gc - 1;
        p[index(i, jo, k, size)] = p[index(i, ji, k, size)];
    }}

    /** y+ grad = 0 */
    for (Int i = gc; i < size[0] - gc; i ++) {
    for (Int k = gc; k < size[2] - gc; k ++) {
        Int ji = size[1] - gc - 1;
        Int jo = size[1] - gc;
        p[index(i, jo, k, size)] = p[index(i, ji, k, size)];
    }}

    /** z- grad = 0 */
    for (Int i = gc; i < size[0] - gc; i ++) {
    for (Int j = gc; j < size[1] - gc; j ++) {
        Int ki = gc;
        Int ko = gc - 1;
        p[index(i, j, ko, size)] = p[index(i, j, ki, size)];
    }}

    /** z+ grad = 0 */
    for (Int i = gc; i < size[0] - gc; i ++) {
    for (Int j = gc; j < size[1] - gc; j ++) {
        Int ki = size[2] - gc - 1;
        Int ko = size[2] - gc;
        p[index(i, j, ko, size)] = p[index(i, j, ki, size)];
    }}
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

        printf("RUNTIME INFO\n");
        printf("\tmax step = %ld\n", this->max_step);
        printf("\tdt = %lf\n", this->dt);
    }
};

struct Mesh {
    Real *x, *y, *z, *dx, *dy, *dz;

    void initialize(string path, Int size[3], Int gc, MpiInfo *mpi) {
        build_mesh(path, x, y, z, dx, dy, dz, size, gc, mpi);

        printf("MESH INFO\n");
        printf("\tfolder = %s\n", path.c_str());
        printf("\tsize = (%ld %ld %ld)\n", size[0], size[1], size[2]);
        printf("\tguide cell = %ld\n", gc);
    }

    void finalize() {
        delete[] x;
        delete[] y;
        delete[] z;
        delete[] dx;
        delete[] dy;
        delete[] dz;
    }
};

struct Cfd {
    Real (*U)[3], (*Uold)[3];
    Real *p, *nut, *div;
    Real Uin[3];
    Real Re, Cs;

    void initialize(Real Uin[3], Real Re, Real Cs, Int size[3]) {
        this->Uin[0] = Uin[0];
        this->Uin[1] = Uin[1];
        this->Uin[2] = Uin[2];
        this->Re = Re;
        this->Cs = Cs;

        Int len = size[0]*size[1]*size[2];
        U = new Real[len][3];
        Uold = new Real[len][3];
        p = new Real[len];
        nut = new Real[len];
        div = new Real[len];

        printf("CFD INFO\n");
        printf("\tRe = %lf\n", this->Re);
        printf("\tCs = %lf\n", this->Cs);
        printf("\tUin = (%lf %lf %lf)\n", this->Uin[0], this->Uin[1], this->Uin[2]);
    }

    void finalize() {
        delete[] U;
        delete[] Uold;
        delete[] p;
        delete[] nut;
        delete[] div;
    }
};

struct Eq {
    Real (*A)[7];
    Real *b, *r;
    Int it, max_it;
    Real err, tol;

    void initialize(Int max_it, Real tol, Int size[3]) {
        this->max_it = max_it;
        this->tol = tol;

        Int len = size[0]*size[1]*size[2];
        A = new Real[len][7];
        b = new Real[len];
        r = new Real[len];

        printf("EQ INFO\n");
        printf("\tmax iteration = %ld\n", this->max_it);
        printf("\ttolerance = %lf\n", this->tol);
    }

    void finalize() {
        delete[] A;
        delete[] b;
        delete[] r;
    }
};

struct Solver {
    Int size[3];
    Int gc = 2;

    MpiInfo mpi;
    Runtime rt;
    Mesh mesh;
    Cfd cfd;
    Eq eq;

    void initialize(string path) {
        printf("SETUP INFO\n");
        printf("\tpath %s\n", path.c_str());

        ifstream setup_file(path);
        auto setup_json = json::parse(setup_file);

        auto &rt_json = setup_json["runtime"];
        Real dt = rt_json["dt"];
        Real total_time = rt_json["time"];
        rt.initialize(total_time/dt, dt);

        auto &mesh_json = setup_json["mesh"];
        string mesh_path = mesh_json["path"];
        mesh.initialize(mesh_path, size, gc, &mpi);

        auto &inflow_json = setup_json["inflow"];
        auto &cfd_json = setup_json["cfd"];
        Real Uin[3];
        Uin[0] = inflow_json["value"][0];
        Uin[1] = inflow_json["value"][1];
        Uin[2] = inflow_json["value"][2];
        Real Re = cfd_json["Re"];
        Real Cs = cfd_json["Cs"];
        cfd.initialize(Uin, Re, Cs, size);

        auto &eq_json = setup_json["eq"];
        Real tol = eq_json["tolerance"];
        Real max_it = eq_json["max_iteration"];
        eq.initialize(max_it, tol, size);
    }

    void finalize() {
        mesh.finalize();
        cfd.finalize();
        eq.finalize();
    }
};

int main(int argc, char *argv[]) {
    Solver solver;
    string setup_path(argv[1]);
    solver.initialize(setup_path);

    write_mesh(
        "data/mesh.txt",
        solver.mesh.x,
        solver.mesh.y,
        solver.mesh.z,
        solver.mesh.dx,
        solver.mesh.dy,
        solver.mesh.dz,
        solver.size,
        solver.gc
    );

    solver.finalize();
}