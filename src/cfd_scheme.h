#pragma once

#include <cmath>

static double calc_convection_term(double stencil[], double U[], double D[]) {
    double &valc  = stencil[0];
    double &vale  = stencil[1];
    double &valee = stencil[2];
    double &valw  = stencil[3];
    double &valww = stencil[4];
    double &valn  = stencil[5];
    double &valnn = stencil[6];
    double &vals  = stencil[7];
    double &valss = stencil[8];
    double &valt  = stencil[9];
    double &valtt = stencil[10];
    double &valb  = stencil[11];
    double &valbb = stencil[12];
    double &u     = U[0];
    double &v     = U[1];
    double &w     = U[2];
    double &dx    = D[0];
    double &dy    = D[1];
    double &dz    = D[2];

    double convection = 0;
    convection += u*(- valee + 8*vale - 8*valw + valww)/(12*dx);
    convection += fabs(u)*(valee - 4*vale + 6*valc - 4*valw + valww)/(12*dx);
    convection += v*(- valnn + 8*valn - 8*vals + valss)/(12*dy);
    convection += fabs(v)*(valnn - 4*valn + 6*valc - 4*vals + valss)/(12*dy);
    convection += w*(- valtt + 8*valt - 8*valb + valbb)/(12*dz);
    convection += fabs(w)*(valtt - 4*valt + 6*valc - 4*valb + valbb)/(12*dz);
    return convection;
}

static double calc_diffusion_term(double stencil[], double D[], double viscosity) {
    double &valc = stencil[0];
    double &vale = stencil[1];
    double &valw = stencil[2];
    double &valn = stencil[3];
    double &vals = stencil[4];
    double &valt = stencil[5];
    double &valb = stencil[6];
    double txc   = 1./D[0];
    double txe   = 1./D[1];
    double txw   = 1./D[2];
    double tyc   = 1./D[3];
    double tyn   = 1./D[4];
    double tys   = 1./D[5];
    double tzc   = 1./D[6];
    double tzt   = 1./D[7];
    double tzb   = 1./D[8];

    double diffusion = 0;
    diffusion += txc*(txc*(vale - 2*valc + valw) + 0.25*(txe - txw)*(vale - valw));
    diffusion += tyc*(tyc*(valn - 2*valc + vals) + 0.25*(tyn - tys)*(valn - vals));
    diffusion += tzc*(tzc*(valt - 2*valc + valb) + 0.25*(tzt - tzb)*(valt - valb));
    diffusion *= viscosity;
    return diffusion;
}
