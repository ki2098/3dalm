#include <cmath>
#include "json.hpp"
#include "io.h"
#include "util.h"
#include "type.h"

using namespace std;
using json = nlohmann::json;

Real calcConvection(
    Real stencil[13],
    Real U[3],
    Real D[3]
) {
    Real valC0 = stencil[0];
    Real valE1 = stencil[1];
    Real valE2 = stencil[2];
    Real valW1 = stencil[3];
    Real valW2 = stencil[4];
    Real valN1 = stencil[5];
    Real valN2 = stencil[6];
    Real valS1 = stencil[7];
    Real valS2 = stencil[8];
    Real valT1 = stencil[9];
    Real valT2 = stencil[10];
    Real valB1 = stencil[11];
    Real valB2 = stencil[12];
    Real u = U[0];
    Real v = U[1];
    Real w = U[2];
    Real dx = D[0];
    Real dy = D[1];
    Real dz = D[2];

    double convection = 0;
    convection += u*(- valE2 + 8*valE1 - 8*valW1 + valW2)/(12*dx);
    convection += fabs(u)*(valE2 - 4*valE1 + 6*valC0 - 4*valW1 + valW2)/(12*dx);
    convection += v*(- valN2 + 8*valN1 - 8*valS1 + valS2)/(12*dy);
    convection += fabs(v)*(valN2 - 4*valN1 + 6*valC0 - 4*valS1 + valS2)/(12*dy);
    convection += w*(- valT2 + 8*valT1 - 8*valB1 + valB2)/(12*dz);
    convection += fabs(w)*(valT2 - 4*valT1 + 6*valC0 - 4*valB1 + valB2)/(12*dy);
    return convection;
}

Real calcDiffusion(
    Real stencil[7],
    Real X[9],
    Real D[3],
    Real viscosity
) {
    Real valC = stencil[0];
    Real valE = stencil[1];
    Real valW = stencil[2];
    Real valN = stencil[3];
    Real valS = stencil[4];
    Real valT = stencil[5];
    Real valB = stencil[6];
    Real xC = X[0];
    Real xE = X[1];
    Real xW = X[2];
    Real yC = X[3];
    Real yN = X[4];
    Real yS = X[5];
    Real zC = X[6];
    Real zT = X[7];
    Real zB = X[8];
    Real dx = D[0];
    Real dy = D[1];
    Real dz = D[2];

    Real diffusion = 0;
    diffusion += ((valE - valC)/(xE - xC) - (valC - valW)/(xC - xW))/dx;
    diffusion += ((valN - valC)/(yN - yC) - (valC - valS)/(yC - yS))/dy;
    diffusion += ((valT - valC)/(zT - zC) - (valC - valB)/(zC - zB))/dz;
    diffusion *= viscosity;
    return diffusion;
}

void calcPseudoU(
    Real U[][3],
    Real UPrev[][3],
    Real nut[],
    Real Re,
    Real dt,
    Real x[],
    Real y[],
    Real z[],
    Real dx[],
    Real dy[],
    Real dz[],
    Int sz[3],
    Int gc,
    MpiInfo *mpi
) {
    Int len = sz[0]*sz[1]*sz[2];

#pragma acc parallel present(U[:len], UPrev[:len], nut[:len], x[:sz[0]], y[:sz[1]], z[:sz[2]], dx[:sz[0]], dy[:sz[1]], dz[:sz[2]]) firstprivate(sz[:3], gc, Re, dt)
#pragma acc loop independent collapse(3)
    for (Int i = gc; i < sz[0] - gc; i ++) {
    for (Int j = gc; j < sz[1] - gc; j ++) {
    for (Int k = gc; k < sz[2] - gc; k ++) {
        Int idC0 = getId(i, j, k, sz);
        Int idE1 = getId(i + 1, j, k, sz);
        Int idE2 = getId(i + 2, j, k, sz);
        Int idW1 = getId(i - 1, j, k, sz);
        Int idW2 = getId(i - 2, j, k, sz);
        Int idN1 = getId(i, j + 1, k, sz);
        Int idN2 = getId(i, j + 2, k, sz);
        Int idS1 = getId(i, j - 1, k, sz);
        Int idS2 = getId(i, j - 2, k, sz);
        Int idT1 = getId(i, j, k + 1, sz);
        Int idT2 = getId(i, j, k + 2, sz);
        Int idB1 = getId(i, j, k - 1, sz);
        Int idB2 = getId(i, j, k - 2, sz);

        Real D[] = {dx[i], dy[j], dz[k]};
        Real XStencil[] = {
            x[i], x[i + 1], x[i - 1],
            y[j], y[j + 1], y[j - 1],
            z[k], z[k + 1], z[k - 1]
        };
        Real viscosity = 1/Re + nut[idC0];

        for (Int m = 0; m < 3; m ++) {
            Real convectionStencil[] = {
                UPrev[idC0][m],
                UPrev[idE1][m],
                UPrev[idE2][m],
                UPrev[idW1][m],
                UPrev[idW2][m],
                UPrev[idN1][m],
                UPrev[idN2][m],
                UPrev[idS1][m],
                UPrev[idS2][m],
                UPrev[idT1][m],
                UPrev[idT2][m],
                UPrev[idB1][m],
                UPrev[idB2][m]
            };
            Real convection = calcConvection(convectionStencil, UPrev[idC0], D);

            Real diffusionStencil[] = {
                UPrev[idC0][m],
                UPrev[idE1][m],
                UPrev[idW1][m],
                UPrev[idN1][m],
                UPrev[idS1][m],
                UPrev[idT1][m],
                UPrev[idB1][m]
            };
            Real diffusion = calcDiffusion(diffusionStencil, XStencil, D, viscosity);

            U[idC0][m] = UPrev[idC0][m] + dt*(- convection + diffusion);
        }
    }}}
}

void calcPoissonRhs(
    Real U[][3],
    Real rhs[],
    Real dt,
    Real scale,
    Real x[],
    Real y[],
    Real z[],
    Int sz[3],
    Int gc,
    MpiInfo *mpi
) {
    Int len = sz[0]*sz[1]*sz[2];

#pragma acc parallel present(U[:len], rhs[:len], x[:sz[0]], y[:sz[1]], z[:sz[2]]) firstprivate(sz[:3], gc, dt, scale)
#pragma acc loop independent collapse(3)
    for (Int i = gc; i < sz[0] - gc; i ++) {
    for (Int j = gc; j < sz[1] - gc; j ++) {
    for (Int k = gc; k < sz[2] - gc; k ++) {
        Int idC = getId(i, j, k, sz);
        Int idE = getId(i + 1, j, k, sz);
        Int idW = getId(i - 1, j, k, sz);
        Int idN = getId(i, j + 1, k, sz);
        Int idS = getId(i, j - 1, k, sz);
        Int idT = getId(i, j, k + 1, sz);
        Int idB = getId(i, j, k - 1, sz);
        double divergence = 0;
        divergence += (U[idE][0] - U[idW][0])/(x[i + 1] - x[i - 1]);
        divergence += (U[idN][1] - U[idS][1])/(y[j + 1] - y[j - 1]);
        divergence += (U[idT][2] - U[idB][2])/(z[k + 1] - z[k - 1]);
        rhs[idC] = divergence/(dt*scale);
    }}}
}

void projectP(
    Real p[],
    Real U[][3],
    Real dt,
    Real x[],
    Real y[],
    Real z[],
    Int sz[3],
    Int gc,
    MpiInfo *mpi
) {
    Int len = sz[0]*sz[1]*sz[2];

#pragma acc parallel present(p[:len], U[:len], x[:sz[0]], y[:sz[1]], z[:sz[2]]) firstprivate(sz[:3], gc, dt)
#pragma acc loop independent collapse(3)
    for (Int i = gc; i < sz[0] - gc; i ++) {
    for (Int j = gc; j < sz[1] - gc; j ++) {
    for (Int k = gc; k < sz[2] - gc; k ++) {
        Int idC = getId(i, j, k, sz);
        Int idE = getId(i + 1, j, k, sz);
        Int idW = getId(i - 1, j, k, sz);
        Int idN = getId(i, j + 1, k, sz);
        Int idS = getId(i, j - 1, k, sz);
        Int idT = getId(i, j, k + 1, sz);
        Int idB = getId(i, j, k - 1, sz);

        U[idC][0] -= dt*(p[idE] - p[idW])/(x[i + 1] - x[i - 1]);
        U[idC][1] -= dt*(p[idN] - p[idS])/(y[j + 1] - y[j - 1]);
        U[idC][2] -= dt*(p[idT] - p[idB])/(z[k + 1] - z[k - 1]);
    }}}
}

void calcNut(
    Real U[][3],
    Real nut[],
    Real Cs,
    Real x[],
    Real y[],
    Real z[],
    Real dx[],
    Real dy[],
    Real dz[],
    Int sz[3],
    Int gc,
    MpiInfo *mpi
) {
    Int len = sz[0]*sz[1]*sz[2];

#pragma acc parallel present(U[:len], nut[:len], x[:sz[0]], y[:sz[1]], z[:sz[2]], dx[:sz[0]], dy[:sz[1]], dz[:sz[2]]) firstprivate(sz[:3], gc, Cs)
#pragma acc loop independent collapse(3)
    for (Int i = gc; i < sz[0] - gc; i ++) {
    for (Int j = gc; j < sz[1] - gc; j ++) {
    for (Int k = gc; k < sz[2] - gc; k ++) {
        Int idC = getId(i, j, k, sz);
        Int idE = getId(i + 1, j, k, sz);
        Int idW = getId(i - 1, j, k, sz);
        Int idN = getId(i, j + 1, k, sz);
        Int idS = getId(i, j - 1, k, sz);
        Int idT = getId(i, j, k + 1, sz);
        Int idB = getId(i, j, k - 1, sz);

        Real dxEW = x[i + 1] - x[i - 1];
        Real dyNS = y[j + 1] - y[j - 1];
        Real dzTB = z[k + 1] - z[k - 1];
        Real volume = dx[i]*dy[j]*dz[k];
        
        Real dudx = (U[idE][0] - U[idW][0])/dxEW;
        Real dudy = (U[idN][0] - U[idS][0])/dyNS;
        Real dudz = (U[idT][0] - U[idB][0])/dzTB;
        Real dvdx = (U[idE][1] - U[idW][1])/dxEW;
        Real dvdy = (U[idN][1] - U[idS][1])/dyNS;
        Real dvdz = (U[idT][1] - U[idB][1])/dzTB;
        Real dwdx = (U[idE][2] - U[idW][2])/dxEW;
        Real dwdy = (U[idN][2] - U[idS][2])/dyNS;
        Real dwdz = (U[idT][2] - U[idB][2])/dzTB;

        Real s1 = 2*square(dudx);
        Real s2 = 2*square(dvdy);
        Real s3 = 2*square(dwdz);
        Real s4 = square(dudy + dvdx);
        Real s5 = square(dudz + dwdx);
        Real s6 = square(dvdz + dwdy);
        Real stress = sqrt(s1 + s2 + s3 + s4 + s5 + s6);
        Real filter = cbrt(volume);
        nut[idC] = square(Cs*filter)*stress;
    }}}
}

void calcDivergence(
    Real U[][3],
    Real div[],
    Real x[],
    Real y[],
    Real z[],
    const Int sz[3],
    Int gc,
    MpiInfo *mpi
) {
    Int len = sz[0]*sz[1]*sz[2];

#pragma acc parallel firstprivate(sz[:3], gc) present(U[:len], div[:len], x[:sz[0]], y[:sz[1]], z[:sz[2]])
#pragma acc loop independent collapse(3)
    for (Int i = gc; i < sz[0] - gc; i ++) {
    for (Int j = gc; j < sz[1] - gc; j ++) {
    for (Int k = gc; k < sz[2] - gc; k ++) {
        Int idC = getId(i    , j    , k    , sz);
        Int idE = getId(i + 1, j    , k    , sz);
        Int idW = getId(i - 1, j    , k    , sz);
        Int idN = getId(i    , j + 1, k    , sz);
        Int idS = getId(i    , j - 1, k    , sz);
        Int idT = getId(i    , j    , k + 1, sz);
        Int idB = getId(i    , j    , k - 1, sz);
        
        Real divergence = 0;
        divergence += (U[idE][0] - U[idW][0])/(x[i + 1] - x[i - 1]);
        divergence += (U[idN][1] - U[idS][1])/(y[j + 1] - y[j - 1]);
        divergence += (U[idT][2] - U[idB][2])/(z[k + 1] - z[k - 1]);
        div[idC] = divergence;
        // if (fabs(divergence) > 1e-3) {
        //     printf("%ld %ld %ld %e %e %e %e %e %e %e\n", i, j, k, U[idE][0], U[idW][0], U[idN][1], U[idS][1], U[idT][2], U[idB][2], divergence);
        // }
    }}}
}

Real calcL2Norm(
    Real v[],
    Int sz[3],
    Int gc
) {
    Int len = sz[0]*sz[1]*sz[2];
    Real total = 0;

#pragma acc parallel present(v[:len]) firstprivate(sz[:3], gc)
#pragma acc loop independent collapse(3) reduction(+:total)
    for (Int i = gc; i < sz[0] - gc; i ++) {
    for (Int j = gc; j < sz[1] - gc; j ++) {
    for (Int k = gc; k < sz[2] - gc; k ++) {
        total += square(v[getId(i, j, k, sz)]);
    }}}
    return sqrt(total);
}

void calcResidual(
    Real A[][7],
    Real x[],
    Real b[],
    Real r[],
    Int sz[3],
    Int gc,
    MpiInfo *mpi
) {
    Int len = sz[0]*sz[1]*sz[2];

#pragma acc parallel present(A[:len], x[:len], b[:len], r[:len]) firstprivate(sz[:3], gc)
#pragma acc loop independent collapse(3)
    for (Int i = gc; i < sz[0] - gc; i ++) {
    for (Int j = gc; j < sz[1] - gc; j ++) {
    for (Int k = gc; k < sz[2] - gc; k ++) {
        Int idC = getId(i, j, k, sz);
        Int idE = getId(i + 1, j, k, sz);
        Int idW = getId(i - 1, j, k, sz);
        Int idN = getId(i, j + 1, k, sz);
        Int idS = getId(i, j - 1, k, sz);
        Int idT = getId(i, j, k + 1, sz);
        Int idB = getId(i, j, k - 1, sz);

        Real aC = A[idC][0];
        Real aE = A[idC][1];
        Real aW = A[idC][2];
        Real aN = A[idC][3];
        Real aS = A[idC][4];
        Real aT = A[idC][5];
        Real aB = A[idC][6];

        Real xC = x[idC];
        Real xE = x[idE];
        Real xW = x[idW];
        Real xN = x[idN];
        Real xS = x[idS];
        Real xT = x[idT];
        Real xB = x[idB];

        r[idC] = b[idC] - (aC*xC + aE*xE + aW*xW + aN*xN + aS*xS + aT*xT + aB*xB);
    }}}
}

void sweepSor(
    Real A[][7],
    Real x[],
    Real b[],
    Real relaxRate,
    Int color,
    Int sz[3],
    Int gc,
    MpiInfo *mpi
) {
    Int len = sz[0]*sz[1]*sz[2];

#pragma acc parallel present(A[:len], x[:len], b[:len]) firstprivate(relaxRate, color, sz[:3], gc)
#pragma acc loop independent collapse(3)
    for (Int i = gc; i < sz[0] - gc; i ++) {
    for (Int j = gc; j < sz[1] - gc; j ++) {
    for (Int k = gc; k < sz[2] - gc; k ++) {
        if ((i + j + k)%2 == color) {
            Int idC = getId(i, j, k, sz);
            Int idE = getId(i + 1, j, k, sz);
            Int idW = getId(i - 1, j, k, sz);
            Int idN = getId(i, j + 1, k, sz);
            Int idS = getId(i, j - 1, k, sz);
            Int idT = getId(i, j, k + 1, sz);
            Int idB = getId(i, j, k - 1, sz);

            Real aC = A[idC][0];
            Real aE = A[idC][1];
            Real aW = A[idC][2];
            Real aN = A[idC][3];
            Real aS = A[idC][4];
            Real aT = A[idC][5];
            Real aB = A[idC][6];

            Real xC = x[idC];
            Real xE = x[idE];
            Real xW = x[idW];
            Real xN = x[idN];
            Real xS = x[idS];
            Real xT = x[idT];
            Real xB = x[idB];

            x[idC] = xC + relaxRate*(b[idC] - (aC*xC + aE*xE + aW*xW + aN*xN + aS*xS + aT*xT + aB*xB))/aC;
        }
    }}}
}

void runSor(
    Real A[][7],
    Real x[],
    Real b[],
    Real r[],
    Real relaxRate,
    Int &it,
    Int maxIt,
    Real &err,
    Real maxErr,
    Int sz[3],
    Int gc,
    MpiInfo *mpi
) {
    Int effectiveCount = (sz[0] - 2*gc)*(sz[1] - 2*gc)*(sz[2] - 2*gc);
    it = 0;
    do {
        sweepSor(A, x, b, relaxRate, 0, sz, gc, mpi);
        sweepSor(A, x, b, relaxRate, 1, sz, gc, mpi);
        calcResidual(A, x, b, r, sz, gc, mpi);
        err = calcL2Norm(r, sz, gc)/sqrt(effectiveCount);
        it ++;
    } while (it < maxIt && err > maxErr);
}

Real prepareA(
    Real A[][7],
    Real x[],
    Real y[],
    Real z[],
    Real dx[],
    Real dy[],
    Real dz[],
    Int sz[3],
    Int gc,
    MpiInfo *mpi
) {
    Int len = sz[0]*sz[1]*sz[2]; 
    Real maxDiag = 0;

#pragma acc parallel present(A[:len], x[:sz[0]], y[:sz[1]], z[:sz[2]], dx[:sz[0]], dy[:sz[1]], dz[:sz[2]]) firstprivate(sz[:3], gc)
#pragma acc loop independent collapse(3) reduction(max:maxDiag)
    for (Int i = gc; i < sz[0] - gc; i ++) {
    for (Int j = gc; j < sz[1] - gc; j ++) {
    for (Int k = gc; k < sz[2] - gc; k ++) {
        Int id = getId(i, j, k, sz);
        Real dxC = dx[i];
        Real dyC = dy[j];
        Real dzC = dz[k];
        Real dxEC = x[i + 1] - x[i];
        Real dxCW = x[i] - x[i - 1];
        Real dyNC = y[j + 1] - y[j];
        Real dyCS = y[j] - y[j - 1];
        Real dzTC = z[k + 1] - z[k];
        Real dzCB = z[k] - z[k - 1];
        Real aE = 1/(dxC*dxEC);
        Real aW = 1/(dxC*dxCW);
        Real aN = 1/(dyC*dyNC);
        Real aS = 1/(dyC*dyCS);
        Real aT = 1/(dzC*dzTC);
        Real aB = 1/(dzC*dzCB);
        Real aC = - (aE + aW + aN + aS + aT + aB);
        A[id][0] = aC;
        A[id][1] = aE;
        A[id][2] = aW;
        A[id][3] = aN;
        A[id][4] = aS;
        A[id][5] = aT;
        A[id][6] = aB;
        if (fabs(aC) > maxDiag) {
            maxDiag = fabs(aC);
        }
    }}}

#pragma acc parallel present(A[:len]) firstprivate(maxDiag, len)
#pragma acc loop independent collapse(2)
    for (Int i = 0; i < len; i ++) {
        for (Int m = 0; m < 7; m ++) {
            A[i][m] /= maxDiag;
        }
    }

    return maxDiag;
}

void applyUbc(
    Real U[][3],
    Real UPrev[][3],
    Real UIn[3],
    Real dt,
    Real x[],
    Real dy[],
    Real dz[],
    Int sz[3],
    Int gc,
    MpiInfo *mpi
) {
    Int len = sz[0]*sz[1]*sz[2]; 

    /** x- fixed value inflow */
#pragma acc parallel present(U[:len]) copyin(UIn[:3]) firstprivate(sz[:3], gc)
#pragma acc loop independent collapse(3)
    for (Int i = 0 ; i < gc        ; i ++) {
    for (Int j = gc; j < sz[1] - gc; j ++) {
    for (Int k = gc; k < sz[2] - gc; k ++) {
        Int idC = getId(i, j, k, sz);
        for (Int m = 0; m < 3; m ++) {
            U[idC][m] = UIn[m];
        }
    }}}

    /** x+ convective outflow */
#pragma acc parallel present(U[:len], UPrev[:len], x[:sz[0]]) firstprivate(sz[:3], gc, dt)
#pragma acc loop independent collapse(3)
    for (Int i = sz[0] - gc; i < sz[0]     ; i ++) {
    for (Int j = gc        ; j < sz[1] - gc; j ++) {
    for (Int k = gc        ; k < sz[2] - gc; k ++) {
        Int id0 = getId(i    , j, k, sz);
        Int id1 = getId(i - 1, j, k, sz);
        Int id2 = getId(i - 2, j, k, sz);
        Real h1 = x[i] - x[i - 1];
        Real h2 = x[i] - x[i - 2];
        Real uOut = UPrev[id0][0];
        for (Int m = 0; m < 3; m ++) {
            Real f0 = UPrev[id0][m];
            Real f1 = UPrev[id1][m];
            Real f2 = UPrev[id2][m];
            Real grad = (f0*(h2*h2 - h1*h1) - f1*h2*h2 + f2*h1*h1)/(h1*h2*h2 - h2*h1*h1);
            U[id0][m] = f0 - uOut*dt*grad;
        }
    }}}

    /** y- slip */
#pragma acc parallel present(U[:len], dy[:sz[1]]) firstprivate(sz[:3], gc)
#pragma acc loop independent collapse(2)
    for (Int i = gc; i < sz[0] - gc; i ++) {
    for (Int k = gc; k < sz[2] - gc; k ++) {
        Int jI1 = gc;
        Int jI2 = gc + 1;
        Int jO1 = gc - 1;
        Int jO2 = gc - 2;
        Int idI1 = getId(i, jI1, k, sz);
        Int idI2 = getId(i, jI2, k, sz);
        Int idO1 = getId(i, jO1, k, sz);
        Int idO2 = getId(i, jO2, k, sz);
        Real UBc[] = {U[idI1][0], 0, U[idI1][2]};
        Real hI1 = 0.5*dy[jI1];
        Real hI2 = 0.5*dy[jI2] + dy[jI1];
        Real hO1 = 0.5*dy[jO1];
        Real hO2 = 0.5*dy[jO2] + dy[jO1];
        for (Int m = 0; m < 3; m ++) {
            U[idO1][m] = UBc[m] - (U[idI1][m] - UBc[m])*(hO1/hI1);
            U[idO2][m] = UBc[m] - (U[idI2][m] - UBc[m])*(hO2/hI2);
        }
    }}

    /** y+ slip */
#pragma acc parallel present(U[:len], dy[:sz[1]]) firstprivate(sz[:3], gc)
#pragma acc loop independent collapse(2)
    for (Int i = gc; i < sz[0] - gc; i ++) {
    for (Int k = gc; k < sz[2] - gc; k ++) {
        Int jI1 = sz[1] - gc - 1;
        Int jI2 = sz[1] - gc - 2;
        Int jO1 = sz[1] - gc;
        Int jO2 = sz[1] - gc + 1;
        Int idI1 = getId(i, jI1, k, sz);
        Int idI2 = getId(i, jI2, k, sz);
        Int idO1 = getId(i, jO1, k, sz);
        Int idO2 = getId(i, jO2, k, sz);
        Real UBc[] = {U[idI1][0], 0, U[idI1][2]};
        Real hI1 = 0.5*dy[jI1];
        Real hI2 = 0.5*dy[jI2] + dy[jI1];
        Real hO1 = 0.5*dy[jO1];
        Real hO2 = 0.5*dy[jO2] + dy[jO1];
        for (Int m = 0; m < 3; m ++) {
            U[idO1][m] = UBc[m] - (U[idI1][m] - UBc[m])*(hO1/hI1);
            U[idO2][m] = UBc[m] - (U[idI2][m] - UBc[m])*(hO2/hI2);
        }
    }}

    /** z- no slip U=0 */
#pragma acc parallel present(U[:len], dz[:sz[2]]) firstprivate(sz[:3], gc)
#pragma acc loop independent collapse(2)
    for (Int i = gc; i < sz[0] - gc; i ++) {
    for (Int j = gc; j < sz[1] - gc; j ++) {
        Int kI1 = gc;
        Int kI2 = gc + 1;
        Int kO1 = gc - 1;
        Int kO2 = gc - 2;
        Int idI1 = getId(i, j, kI1, sz);
        Int idI2 = getId(i, j, kI2, sz);
        Int idO1 = getId(i, j, kO1, sz);
        Int idO2 = getId(i, j, kO2, sz);
        Real UBc[] = {0, 0, 0};
        Real hI1 = 0.5*dz[kI1];
        Real hI2 = 0.5*dz[kI2] + dz[kI1];
        Real hO1 = 0.5*dz[kO1];
        Real hO2 = 0.5*dz[kO2] + dz[kO1];
        for (Int m = 0; m < 3; m ++) {
            U[idO1][m] = UBc[m] - (U[idI1][m] - UBc[m])*(hO1/hI1);
            U[idO2][m] = UBc[m] - (U[idI2][m] - UBc[m])*(hO2/hI2);
        }
    }}

    /** z+ slip */
#pragma acc parallel present(U[:len], dz[:sz[2]]) firstprivate(sz[:3], gc)
#pragma acc loop independent collapse(2)
    for (Int i = gc; i < sz[0] - gc; i ++) {
    for (Int j = gc; j < sz[1] - gc; j ++) {
        Int kI1 = sz[2] - gc - 1;
        Int kI2 = sz[2] - gc - 2;
        Int kO1 = sz[2] - gc;
        Int kO2 = sz[2] - gc + 1;
        Int idI1 = getId(i, j, kI1, sz);
        Int idI2 = getId(i, j, kI2, sz);
        Int idO1 = getId(i, j, kO1, sz);
        Int idO2 = getId(i, j, kO2, sz);
        Real UBc[] = {U[idI1][0], U[idI1][1], 0};
        Real hI1 = 0.5*dz[kI1];
        Real hI2 = 0.5*dz[kI2] + dz[kI1];
        Real hO1 = 0.5*dz[kO1];
        Real hO2 = 0.5*dz[kO2] + dz[kO1];
        for (Int m = 0; m < 3; m ++) {
            U[idO1][m] = UBc[m] - (U[idI1][m] - UBc[m])*(hO1/hI1);
            U[idO2][m] = UBc[m] - (U[idI2][m] - UBc[m])*(hO2/hI2);
        }
    }}
}

void applyPBc(
    Real p[],
    Real dx[],
    Int sz[3],
    Int gc,
    MpiInfo *mpi
) {
    Int len = sz[0]*sz[1]*sz[2];

    /** x- gradient = 0 */
#pragma acc parallel present(p[:len]) firstprivate(sz[:3], gc)
#pragma acc loop independent collapse(2)
    for (Int j = gc; j < sz[1] - gc; j ++) {
    for (Int k = gc; k < sz[2] - gc; k ++) {
        Int iI = gc;
        Int iO = gc - 1;
        p[getId(iO, j, k, sz)] = p[getId(iI, j, k, sz)];
    }}

    /** x+ fixed value = 0 */
#pragma acc parallel present(p[:len], dx[:sz[0]]) firstprivate(sz[:3], gc)
#pragma acc loop independent collapse(2)
    for (Int j = gc; j < sz[1] - gc; j ++) {
    for (Int k = gc; k < sz[2] - gc; k ++) {
        Int iI = sz[0] - gc - 1;
        Int iO = sz[0] - gc;
        Real pBc = 0;
        Real hI = 0.5*dx[iI];
        Real hO = 0.5*dx[iO];
        p[getId(iO, j, k, sz)] = pBc - (p[getId(iI, j, k, sz)] - pBc)*(hO/hI);
    }}

    /** y- gradient = 0 */
#pragma acc parallel present(p[:len]) firstprivate(sz[:3], gc)
#pragma acc loop independent collapse(2)
    for (Int i = gc; i < sz[0] - gc; i ++) {
    for (Int k = gc; k < sz[2] - gc; k ++) {
        Int jI = gc;
        Int jO = gc - 1;
        p[getId(i, jO, k, sz)] = p[getId(i, jI, k, sz)];
    }}

    /** y+ gradient = 0 */
#pragma acc parallel present(p[:len]) firstprivate(sz[:3], gc)
#pragma acc loop independent collapse(2)
    for (Int i = gc; i < sz[0] - gc; i ++) {
    for (Int k = gc; k < sz[2] - gc; k ++) {
        Int jI = sz[1] - gc - 1;
        Int jO = sz[1] - gc;
        p[getId(i, jO, k, sz)] = p[getId(i, jI, k, sz)];
    }}

    /** z- gradient = 0 */
#pragma acc parallel present(p[:len]) firstprivate(sz[:3], gc)
#pragma acc loop independent collapse(2)
    for (Int i = gc; i < sz[0] - gc; i ++) {
    for (Int j = gc; j < sz[1] - gc; j ++) {
        Int kI = gc;
        Int kO = gc - 1;
        p[getId(i, j, kO, sz)] = p[getId(i, j, kI, sz)];
    }}

    /** z+ gradient = 0 */
#pragma acc parallel present(p[:len]) firstprivate(sz[:3], gc)
#pragma acc loop independent collapse(2)
    for (Int i = gc; i < sz[0] - gc; i ++) {
    for (Int j = gc; j < sz[1] - gc; j ++) {
        Int kI = sz[2] - gc - 1;
        Int kO = sz[2] - gc;
        p[getId(i, j, kO, sz)] = p[getId(i, j, kI, sz)];
    }}
}

struct Runtime {
    Int step = 0, maxStep;
    Real dt;

    void initialize(Int maxStep, Real dt) {
        this->maxStep = maxStep;
        this->dt = dt;

        printf("RUNTIME INFO\n");
        printf("\tdt = %e\n", this->dt);
        printf("\tmax step = %d\n", this->maxStep);
    }

    Real getTime() {
        return step*dt;
    }
};

struct Mesh {
    Real *x, *y, *z, *dx, *dy, *dz;

    void initialize(string path, Int sz[3], Int gc, MpiInfo *mpi) {
        buildMeshFromDir(path, x, y, z, dx, dy, dz, sz, gc, mpi);

        printf("MESH INFO\n");
        printf("\tpath = %s\n", path.c_str());
        printf("\tsize = (%d %d %d)\n", sz[0], sz[1], sz[2]);
        printf("\tguide cell = %d\n", gc);

#pragma acc enter data copyin(x[:sz[0]], y[:sz[1]], z[:sz[2]], dx[:sz[0]], dy[:sz[1]], dz[:sz[2]])
    }

    void finalize(Int sz[3]) {
        delete[] x;
        delete[] y;
        delete[] z;
        delete[] dx;
        delete[] dy;
        delete[] dz;

#pragma acc exit data delete(x[:sz[0]], y[:sz[1]], z[:sz[2]], dx[:sz[0]], dy[:sz[1]], dz[:sz[2]])
    }
};

struct Cfd {
    Real Re, Cs;
    Real UIn[3];
    Real (*U)[3], (*UPrev)[3];
    Real *p, *nut, *div;
    Real avgDivergence;

    void initialize(Real Re, Real Cs, Real UIn[3], Int sz[3]) {
        this->Re = Re;
        this->Cs = Cs;
        this->UIn[0] = UIn[0];
        this->UIn[1] = UIn[1];
        this->UIn[2] = UIn[2];
        Int len = sz[0]*sz[1]*sz[2];
        U = new Real[len][3];
        UPrev = new Real[len][3];
        p = new Real[len];
        nut = new Real[len];
        div = new Real[len];

        printf("CFD INFO\n");
        printf("\tRe = %e\n", this->Re);
        printf("\tCs = %e\n", this->Cs);
        printf("INFLOW INFO\n");
        printf("\tinflow U = (%lf %lf %lf)\n", this->UIn[0], this->UIn[1], this->UIn[2]);

#pragma acc enter data create(U[:len], UPrev[:len], p[:len], nut[:len], div[:len])
    }

    void finalize(Int sz[3]) {
        delete[] U;
        delete[] UPrev;
        delete[] p;
        delete[] nut;
        delete[] div;

        Int len = sz[0]*sz[1]*sz[2];
#pragma acc exit data delete(U[:len], UPrev[:len], p[:len], nut[:len], div[:len])
    }
};

struct Eq {
    Int it = 0, maxIt;
    Real err = 0, maxErr;
    Real maxDiag;
    Real relaxRate = 1.2;
    Real (*A)[7];
    Real *b, *r;

    void initialize(Int maxIt, Real maxErr, Int sz[3]) {
        this->maxIt = maxIt;
        this->maxErr = maxErr;
        Int len = sz[0]*sz[1]*sz[2];
        A = new Real[len][7];
        b = new Real[len];
        r = new Real[len];

        printf("EQ INFO\n");
        printf("\tmax iteration = %d\n", this->maxIt);
        printf("\tmax error = %e\n", this->maxErr);
        printf("\trelax rate = %lf\n", this->relaxRate);

#pragma acc enter data create(A[:len], b[:len], r[:len])
    }

    void finalize(Int sz[3]) {
        delete[] A;
        delete[] b;
        delete[] r;

        Int len = sz[0]*sz[1]*sz[2];
#pragma acc exit data delete(A[:len], b[:len], r[:len])
    }
};

struct Solver {
    Int sz[3];
    Int gc = 2;

    MpiInfo mpi;
    Runtime rt;
    Mesh mesh;
    Cfd cfd;
    Eq eq;

    void initialize(string path) {
        printf("SOLVER INITIALIZE...\n");

        ifstream setupFile(path);
        auto setupJson = json::parse(setupFile);

        auto &rtJson = setupJson["runtime"];
        Real dt = rtJson["dt"];
        Real maxTime = rtJson["time"];
        rt.initialize(maxTime/dt, dt);

        auto &meshJson = setupJson["mesh"];
        string meshPath = meshJson["path"];
        mesh.initialize(meshPath, this->sz, this->gc, &mpi);

        // #pragma acc enter data copyin(sz[:3])

        Int len = sz[0]*sz[1]*sz[2];
        Int effectiveCount = (sz[0] - 2*gc)*(sz[1] - 2*gc)*(sz[2] - 2*gc);

        auto &cfdJson = setupJson["cfd"];
        auto &inflowJson = setupJson["inflow"];
        Real Cs = cfdJson["Cs"];
        Real Re = cfdJson["Re"];
        Real UIn[3];
        UIn[0] = inflowJson["value"][0];
        UIn[1] = inflowJson["value"][1];
        UIn[2] = inflowJson["value"][2];
        cfd.initialize(Re, Cs, UIn, sz);

        auto &eqJson = setupJson["eq"];
        Real maxErr = eqJson["maxError"];
        Int maxIt = eqJson["maxIteration"];
        eq.initialize(maxIt, maxErr, sz);

        eq.maxDiag = prepareA(
            eq.A,
            mesh.x,
            mesh.y,
            mesh.z,
            mesh.dx,
            mesh.dy,
            mesh.dz,
            sz,
            gc,
            &mpi
        );
        printf("max A diag = %lf\n", eq.maxDiag);

        fillArray(cfd.U, cfd.UIn, len);
        fillArray(cfd.p, 0., len);
        cpyArray(cfd.UPrev, cfd.U, len);
        applyUbc(
            cfd.U,
            cfd.UPrev,
            cfd.UIn,
            rt.dt,
            mesh.x,
            mesh.dy,
            mesh.dz,
            sz,
            gc,
            &mpi
        );

        calcDivergence(
            cfd.U,
            cfd.div,
            mesh.x,
            mesh.y,
            mesh.z,
            sz,
            gc,
            &mpi
        );

        cfd.avgDivergence = calcL2Norm(cfd.div, sz, gc)/sqrt(effectiveCount);

        printf("initial div = %e\n", cfd.avgDivergence);

        printf("SOLVER INITIALIZED\n");
    }

    void finalize() {
        mesh.finalize(this->sz);
        cfd.finalize(this->sz);
        eq.finalize(this->sz);
        // #pragma acc exit data delete(sz[:3])
    }

    void main_loop() {
        Int len = sz[0]*sz[1]*sz[2];
        Int effectiveCount = (sz[0] - 2*gc)*(sz[1] - 2*gc)*(sz[2] - 2*gc);

        cpyArray(cfd.UPrev, cfd.U, len);
        // cout << "loop step 1" << endl;

        calcPseudoU(
            cfd.U,
            cfd.UPrev,
            cfd.nut,
            cfd.Re,
            rt.dt,
            mesh.x,
            mesh.y,
            mesh.z,
            mesh.dx,
            mesh.dy,
            mesh.dz,
            sz,
            gc,
            &mpi
        );
        // cout << "loop step 2" << endl;

        calcPoissonRhs(
            cfd.U,
            eq.b,
            rt.dt,
            eq.maxDiag,
            mesh.x,
            mesh.y,
            mesh.z,
            sz,
            gc,
            &mpi
        );
        // cout << "loop step 3" << endl;

        runSor(
            eq.A,
            cfd.p,
            eq.b,
            eq.r,
            eq.relaxRate,
            eq.it,
            eq.maxIt,
            eq.err,
            eq.maxErr,
            sz,
            gc,
            &mpi
        );
        // cout << "loop step 4" << endl;

        applyPBc(
            cfd.p,
            mesh.dx,
            sz,
            gc,
            &mpi
        );
        // cout << "loop step 5" << endl;

        projectP(
            cfd.p,
            cfd.U,
            rt.dt,
            mesh.x,
            mesh.y,
            mesh.z,
            sz,
            gc,
            &mpi
        );
        // cout << "loop step 6" << endl;

        applyUbc(
            cfd.U,
            cfd.UPrev,
            cfd.UIn,
            rt.dt,
            mesh.x,
            mesh.dy,
            mesh.dz,
            sz,
            gc,
            &mpi
        );
        // cout << "loop step 7" << endl;

        calcDivergence(
            cfd.U,
            cfd.div,
            mesh.x,
            mesh.y,
            mesh.z,
            sz,
            gc,
            &mpi
        );
        // cout << "loop step 8" << endl;

        cfd.avgDivergence = calcL2Norm(cfd.div, sz, gc) / sqrt(effectiveCount);
        // cout << "loop step 9" << endl;

        rt.step ++;

        printf("%d %e %d %e %e\n", rt.step, rt.getTime(), eq.it, eq.err, cfd.avgDivergence);
        fflush(stdout);
    }
};

int main(int argc, char *argv[]) {
    Solver solver;
    string setupPath(argv[1]);
    solver.initialize(setupPath);
    Int len = solver.sz[0]*solver.sz[1]*solver.sz[2];

    writeMesh(
        "data/mesh.txt",
        solver.mesh.x,
        solver.mesh.y,
        solver.mesh.z,
        solver.mesh.dx,
        solver.mesh.dy,
        solver.mesh.dz,
        solver.sz,
        solver.gc
    );

    for (; solver.rt.step < 1000;) {
        solver.main_loop();
    }

#pragma acc update host(solver.cfd.U[:len], solver.cfd.p[:len], solver.cfd.div[:len])
    Real *var[] = {solver.cfd.U[0], solver.cfd.p, solver.cfd.div};
    Int varDim[] = {3, 1, 1};
    string varName[] = {"U", "p", "div"};
    writeCsv(
        "data/plain_windtunnel.csv",
        var,
        3,
        varDim,
        varName,
        solver.mesh.x,
        solver.mesh.y,
        solver.mesh.z,
        solver.sz,
        solver.gc
    );

    solver.finalize();
}
