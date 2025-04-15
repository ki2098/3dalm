#include <cmath>
#include "json.hpp"
#include "io.h"
#include "util.h"
#include "type.h"

using namespace std;
using json = nlohmann::json;

Real calcConvection(
    Real13 stencil,
    Real3 U,
    Real3 D
) {
    Real valC  = stencil[0];
    Real valE  = stencil[1];
    Real valEe = stencil[2];
    Real valW  = stencil[3];
    Real valWw = stencil[4];
    Real valN  = stencil[5];
    Real valNn = stencil[6];
    Real valS  = stencil[7];
    Real valSs = stencil[8];
    Real valT  = stencil[9];
    Real valTt = stencil[10];
    Real valB  = stencil[11];
    Real valBb = stencil[12];
    Real u = U[0];
    Real v = U[1];
    Real w = U[2];
    Real dx = D[0];
    Real dy = D[1];
    Real dz = D[2];

    double convection = 0;
    convection += u*(- valEe + 8*valE - 8*valW + valWw)/(12*dx);
    convection += fabs(u)*(valEe - 4*valE + 6*valC - 4*valW + valWw)/(12*dx);
    convection += v*(- valNn + 8*valN - 8*valS + valSs)/(12*dy);
    convection += fabs(v)*(valNn - 4*valN + 6*valC - 4*valS + valSs)/(12*dy);
    convection += w*(- valTt + 8*valT - 8*valB + valBb)/(12*dz);
    convection += fabs(w)*(valTt - 4*valT + 6*valC - 4*valB + valBb)/(12*dy);
    return convection;
}

Real calcDiffusion(
    Real7 stencil,
    Real9 X,
    Real3 D,
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
    Real3 U[],
    Real3 UPrev[],
    Real nut[],
    Real Re,
    Real dt,
    Real x[],
    Real y[],
    Real z[],
    Real dx[],
    Real dy[],
    Real dz[],
    Int3 sz,
    Int gc,
    MpiInfo *mpi
) {
    for (Int i = gc; i < sz[0] - gc; i ++) {
    for (Int j = gc; j < sz[1] - gc; j ++) {
    for (Int k = gc; k < sz[2] - gc; k ++) {
        Int idC  = getId(i, j, k, sz);
        Int idE  = getId(i + 1, j, k, sz);
        Int idEe = getId(i + 2, j, k, sz);
        Int idW  = getId(i - 1, j, k, sz);
        Int idWw = getId(i - 2, j, k, sz);
        Int idN  = getId(i, j + 1, k, sz);
        Int idNn = getId(i, j + 2, k, sz);
        Int idS  = getId(i, j - 1, k, sz);
        Int idSs = getId(i, j - 2, k, sz);
        Int idT  = getId(i, j, k + 1, sz);
        Int idTt = getId(i, j, k + 2, sz);
        Int idB  = getId(i, j, k - 1, sz);
        Int idBb = getId(i, j, k - 2, sz);

        Real3 D = {dx[i], dy[j], sz[k]};
        Real9 XStencil = {
            x[i], x[i + 1], x[i - 1],
            y[j], y[j + 1], y[j - 1],
            z[k], z[k + 1], z[k - 1]
        };
        Real viscosity = 1/Re + nut[idC];

        for (Int m = 0; m < 3; m ++) {
            Real13 convectionStencil = {
                UPrev[idC ][m],
                UPrev[idE ][m],
                UPrev[idEe][m],
                UPrev[idW ][m],
                UPrev[idWw][m],
                UPrev[idN ][m],
                UPrev[idNn][m],
                UPrev[idS ][m],
                UPrev[idSs][m],
                UPrev[idT ][m],
                UPrev[idTt][m],
                UPrev[idB ][m],
                UPrev[idBb][m]
            };
            Real convection = calcConvection(convectionStencil, UPrev[idC], D);

            Real7 diffusionStencil = {
                UPrev[idC ][m],
                UPrev[idE ][m],
                UPrev[idW ][m],
                UPrev[idN ][m],
                UPrev[idS ][m],
                UPrev[idT ][m],
                UPrev[idB ][m]
            };
            Real diffusion = calcDiffusion(diffusionStencil, XStencil, D, viscosity);

            U[idC][m] = UPrev[idC][m] + dt*(- convection + diffusion);
        }
    }}}
}

void calcPoissonRhs(
    Real3 U[],
    Real rhs[],
    Real dt,
    Real scale,
    Real x[],
    Real y[],
    Real z[],
    Int3 sz,
    Int gc,
    MpiInfo *mpi
) {
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
    Real3 U[],
    Real dt,
    Real x[],
    Real y[],
    Real z[],
    Int3 sz,
    Int gc,
    MpiInfo *mpi
) {
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
    Real3 U[],
    Real nut[],
    Real Cs,
    Real x[],
    Real y[],
    Real z[],
    Real dx[],
    Real dy[],
    Real dz[],
    Int3 sz,
    Int gc,
    MpiInfo *mpi
) {
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
    Real3 U[],
    Real div[],
    Real x[],
    Real y[],
    Real z[],
    Int3 sz,
    Int gc,
    MpiInfo *mpi
) {
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
        
        Real divergence = 0;
        divergence += (U[idE][0] - U[idW][0])/(x[i + 1] - x[i - 1]);
        divergence += (U[idN][1] - U[idS][1])/(y[j + 1] - y[j - 1]);
        divergence += (U[idT][2] - U[idB][2])/(z[k + 1] - z[k - 1]);
        div[idC] = divergence;
    }}}
}

Real calcL2Norm(
    Real v[],
    Int len
) {
    Real total = 0;
    for (Int i = 0; i < len; i ++) {
        total += square(v[i]);
    }
    return sqrt(total);
}

void calcResidual(
    Real7 A[],
    Real x[],
    Real b[],
    Real r[],
    Int3 sz,
    Int gc,
    MpiInfo *mpi
) {
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
    Real7 A[],
    Real x[],
    Real b[],
    Real relaxRate,
    Int color,
    Int3 sz,
    Int gc,
    MpiInfo *mpi
) {
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
    Real7 A[],
    Real x[],
    Real b[],
    Real r[],
    Real relaxRate,
    Int &it,
    Int maxIt,
    Real &err,
    Real maxErr,
    Int3 sz,
    Int gc,
    MpiInfo *mpi
) {
    Int cnt = sz[0]*sz[1]*sz[2];
    Int effectiveCnt = (sz[0] - 2*gc)*(sz[1] - 2*gc)*(sz[2] - 2*gc);
    it = 0;
    do {
        sweepSor(A, x, b, relaxRate, 0, sz, gc, mpi);
        sweepSor(A, x, b, relaxRate, 1, sz, gc, mpi);
        calcResidual(A, x, b, r, sz, gc, mpi);
        err = calcL2Norm(r, cnt)/sqrt(effectiveCnt);
        it ++;
    } while (it < maxIt && err > maxErr);
}

Real prepareA(
    Real7 A[],
    Real x[],
    Real y[],
    Real z[],
    Real dx[],
    Real dy[],
    Real dz[],
    Int3 sz,
    Int gc,
    MpiInfo *mpi
) {
    Int cnt = sz[0]*sz[1]*sz[2]; 
    Real maxDiag = 0;

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
        Real aC = - (aE + aW + aN * aS + aT *aB);
        A[id][0] = aC;
        A[id][1] = aE;
        A[id][2] = aW;
        A[id][3] = aN;
        A[id][4] = aS;
        A[id][5] = aT;
        A[id][6] = aB;
        if (fabs(aC) > maxDiag) {
            maxDiag = aC;
        }
    }}}

    for (Int i = 0; i < cnt; i ++) {
        for (Int m = 0; m < 7; m ++) {
            A[i][m] /= maxDiag;
        }
    }

    return maxDiag;
}

void applyUBc(
    Real3 U[],
    Real3 UPrev[],
    Real3 UIn,
    Real dt,
    Real x[],
    Real y[],
    Real z[],
    Real dx[],
    Real dy[],
    Real dz[],
    Int3 sz,
    Int gc,
    MpiInfo *mpi
) {
    /** x- fixed value inflow */
    for (Int i = 0 ; i < gc        ; i ++) {
    for (Int j = gc; j < sz[1] - gc; j ++) {
    for (Int k = gc; k < sz[2] - gc; k ++) {
        Int idC  = getId(i    , j, k, sz);
        for (Int m = 0; m < 3; m ++) {
            U[idC][m] = UIn[m];
        }
    }}}

    /** x+ convective outflow */
    for (Int i = sz[0] - gc; i < sz[0]     ; i ++) {
    for (Int j = gc        ; j < sz[1] - gc; j ++) {
    for (Int k = gc        ; k < sz[2] - gc; k ++) {
        Int id0 = getId(i    , j, k, sz);
        Int id1 = getId(i - 1, j, k, sz);
        Int id2 = getId(i - 2, j, k, sz);
        Real d1 = x[i] - x[i - 1];
        Real d2 = x[i] - x[i - 2];
        Real uOut = UPrev[id0][0];
        for (Int m = 0; m < 3; m ++) {
            Real val0 = UPrev[id0][m];
            Real val1 = UPrev[id1][m];
            Real val2 = UPrev[id2][m];
            Real gradient = (val0*(d2*d2 - d1*d1) - val1*(d2*d2) + val2*(d1*d1))/(d1*d2*d2 - d2*d1*d1);
            U[id0][m] = val0 - uOut*dt*gradient;
        }
    }}}

    /** y- slip */
    for (Int i = gc; i < sz[0] - gc; i ++) {
    for (Int k = gc; k < sz[2] - gc; k ++) {
        Int jI  = gc;
        Int jIi = gc + 1;
        Int jO  = gc - 1;
        Int jOo = gc - 2;
        Int idI  = getId(i, jI , k, sz);
        Int idIi = getId(i, jIi, k, sz);
        Int idO  = getId(i, jO , k, sz);
        Int idOo = getId(i, jOo, k, sz);
        Real3 UB = {U[idI][1], 0, U[idI][2]};
        Real dBI  = 0.5*dy[jI ];
        Real dBIi = 0.5*dy[jIi] + dy[jI];
        Real dBO  = 0.5*dy[jO ];
        Real dBOo = 0.5*dy[jOo] + dy[jO];
        for (Int m = 0; m < 3; m ++) {
            U[idO ][m] = UB[m] - dBO *(U[idI ][m] - UB[m])/dBI ;
            U[idOo][m] = UB[m] - dBOo*(U[idIi][m] - UB[m])/dBIi;
        }
    }}

    /** y+ slip */
    for (Int i = gc; i < sz[0] - gc; i ++) {
    for (Int k = gc; k < sz[2] - gc; k ++) {
        Int jI  = sz[1] - gc - 1;
        Int jIi = sz[1] - gc - 2;
        Int jO  = sz[1] - gc;
        Int jOo = sz[1] - gc + 1;
        Int idI  = getId(i, jI , k, sz);
        Int idIi = getId(i, jIi, k, sz);
        Int idO  = getId(i, jO , k, sz);
        Int idOo = getId(i, jOo, k, sz);
        Real3 UB = {U[idI][1], 0, U[idI][2]};
        Real dBI  = 0.5*dy[jI ];
        Real dBIi = 0.5*dy[jIi] + dy[jI];
        Real dBO  = 0.5*dy[jO ];
        Real dBOo = 0.5*dy[jOo] + dy[jO];
        for (Int m = 0; m < 3; m ++) {
            U[idO ][m] = UB[m] - dBO *(U[idI ][m] - UB[m])/dBI ;
            U[idOo][m] = UB[m] - dBOo*(U[idIi][m] - UB[m])/dBIi;
        }
    }}

    /** z- no slip */
    for (Int i = 0; i < sz[0] - gc; i ++) {
    for (Int j = 0; j < sz[1] - gc; j ++) {
        Int kI  = gc;
        Int kIi = gc + 1;
        Int kO  = gc - 1;
        Int kOo = gc - 2;
        Int idI  = getId(i, j, kI , sz);
        Int idIi = getId(i, j, kIi, sz);
        Int idO  = getId(i, j, kO , sz);
        Int idOo = getId(i, j, kOo, sz);
        Real3 UB = {0, 0, 0};
        Real dBI  = 0.5*dz[kI ];
        Real dBIi = 0.5*dz[kIi] + dy[kI];
        Real dBO  = 0.5*dz[kO ];
        Real dBOo = 0.5*dz[kOo] + dy[kO];
        for (Int m = 0; m < 3; m ++) {
            U[idO ][m] = UB[m] - dBO *(U[idI ][m] - UB[m])/dBI ;
            U[idOo][m] = UB[m] - dBOo*(U[idIi][m] - UB[m])/dBIi;
        }
    }}

    /** z+ slip */
    for (Int i = 0; i < sz[0] - gc; i ++) {
    for (Int j = 0; j < sz[1] - gc; j ++) {
        Int kI  = sz[2] - gc - 1;
        Int kIi = sz[2] - gc - 2;
        Int kO  = sz[2] - gc;
        Int kOo = sz[2] - gc + 1;
        Int idI  = getId(i, j, kI , sz);
        Int idIi = getId(i, j, kIi, sz);
        Int idO  = getId(i, j, kO , sz);
        Int idOo = getId(i, j, kOo, sz);
        Real3 UB = {U[idI][0], U[idI][1], 0};
        Real dBI  = 0.5*dz[kI ];
        Real dBIi = 0.5*dz[kIi] + dy[kI];
        Real dBO  = 0.5*dz[kO ];
        Real dBOo = 0.5*dz[kOo] + dy[kO];
        for (Int m = 0; m < 3; m ++) {
            U[idO ][m] = UB[m] - dBO *(U[idI ][m] - UB[m])/dBI ;
            U[idOo][m] = UB[m] - dBOo*(U[idIi][m] - UB[m])/dBIi;
        }
    }}
}

void applyPBc(
    Real p[],
    Real dx[],
    Int3 sz,
    Int gc,
    MpiInfo *mpi
) {
    /** x- gradient = 0 */
    for (Int j = gc; j < sz[1] - gc; j ++) {
    for (Int k = gc; k < sz[2] - gc; k ++) {
        p[getId(gc - 1, j, k, sz)] = p[getId(gc, j, k, sz)];
    }}

    /** x+ fixed value = 0 */
    for (Int j = gc; j < sz[1] - gc; j ++) {
    for (Int k = gc; k < sz[2] - gc; k ++) {
        Int iI = sz[0] - gc - 1;
        Int iO = sz[0] - gc;
        Real pB = 0;
        p[getId(iO, j, k, sz)] = pB - dx[iO]*(p[getId(iI, j, k, sz)]- pB)/dx[iI];
    }}
}

struct Runtime {
    Int step = 0, maxStep;
    Real dt;

    void initialize(Int maxStep, Real dt) {
        this->maxStep = maxStep;
        this->dt = dt;

        printf("RUNTIME INFO\n");
        printf("\tdt = %e\n", dt);
        printf("\tmax step = %d\n", maxStep);
    }

    Real getTime() {
        return step*dt;
    }
};

struct Mesh {
    Real *x, *y, *z, *dx, *dy, *dz;

    void initialize(string path, Int3 sz, Int gc, MpiInfo *mpi) {
        buildMeshFromDir(path, x, y, z, dx, dy, dz, sz, gc, mpi);

        printf("MESH INFO\n");
        printf("\tpath = %s\n", path.c_str());
        printf("\tsize = (%d %d %d)\n", sz[0], sz[1], sz[2]);
        printf("\tguide cell = %d\n", gc);
    }

    void finalize(Int3 sz) {
        delete[] x;
        delete[] y;
        delete[] z;
        delete[] dx;
        delete[] dy;
        delete[] dz;
    }
};

struct Cfd {
    Real Re, Cs;
    Real3 UIn;
    Real3 *U, *UPrev;
    Real *p, *nut;

    void initialize(Real Re, Real Cs, Real3 UIn, Int3 sz) {
        this->Re = Re;
        this->Cs = Cs;
        this->UIn[0] = UIn[0];
        this->UIn[1] = UIn[1];
        this->UIn[2] = UIn[2];
        Int cnt = sz[0]*sz[1]*sz[2];
        U = new Real3[cnt];
        UPrev = new Real3[cnt];
        p = new Real[cnt];
        nut = new Real[cnt];

        printf("CFD INFO\n");
        printf("\tRe = %e\n", Re);
        printf("\tCs = %e\n", Cs);
        printf("INFLOW INFO\n");
        printf("\tinflow U = (%lf %lf %lf)\n", UIn[0], UIn[1], UIn[2]);
    }

    void finalize(Int3 sz) {
        delete[] U;
        delete[] UPrev;
        delete[] p;
        delete[] nut;
    }
};

struct Eq {
    Int it = 0, maxIt;
    Real err = 0, maxErr;
    Real7 *A;
    Real *b, *r;

    void initialize(Int maxIt, Real maxErr, Int3 sz) {
        this->maxIt = maxIt;
        this->maxErr = maxErr;
        Int cnt = sz[0]*sz[1]*sz[2];
        A = new Real7[cnt];
        b = new Real[cnt];
        r = new Real[cnt];

        printf("EQ INFO\n");
        printf("\tmax iteration = %d\n", maxIt);
        printf("\tmax error = %e\n", maxErr);
    }

    void finalize(Int3 sz) {
        delete[] A;
        delete[] b;
        delete[] r;
    }
};

struct Solver {
    Int3 sz;
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

        auto &cfdJson = setupJson["cfd"];
        auto &inflowJson = setupJson["inflow"];
        Real Cs = cfdJson["Cs"];
        Real Re = cfdJson["Re"];
        Real3 UIn;
        UIn[0] = inflowJson["value"][0];
        UIn[1] = inflowJson["value"][1];
        UIn[2] = inflowJson["value"][2];
        cfd.initialize(Re, Cs, UIn, sz);

        auto &eqJson = setupJson["eq"];
        Real maxErr = eqJson["maxError"];
        Int maxIt = eqJson["maxIteration"];
        eq.initialize(maxIt, maxErr, sz);

        printf("SOLVER INITIALIZED\n");
    }

    void finalize() {
        mesh.finalize(this->sz);
        cfd.finalize(this->sz);
        eq.finalize(this->sz);
    }
};

int main(int argc, char *argv[]) {
    Solver solver;
    string setupPath(argv[1]);
    solver.initialize(setupPath);
    solver.finalize();
}

