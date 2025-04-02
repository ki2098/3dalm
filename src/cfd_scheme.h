#pragma once

#include <cmath>

static double calc_convection_term(double stencil[], double velocity[], double transform[]) {
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
    double &u     = velocity[0];
    double &v     = velocity[1];
    double &w     = velocity[2];
    double &tx    = transform[0];
    double &ty    = transform[1];
    double &tz    = transform[2];

    double convection = 0;
    convection += tx*u*(- valee + 8*vale - 8*valw + valww)/12.;
    convection += tx*fabs(u)*(valee - 4*vale + 6*valc - 4*valw + valww)/12.;
    convection += ty*v*(- valnn + 8*valn - 8*vals + valss)/12.;
    convection += ty*fabs(v)*(valnn - 4*valn + 6*valc - 4*vals + valss)/12.;
    convection += tz*w*(- valtt + 8*valt - 8*valb + valbb)/12.;
    convection += tz*fabs(w)*(valtt - 4*valt + 6*valc - 4*valb + valbb)/12.;
    return convection;
}

static double calc_diffusion_term(double stencil[], double transform[], double viscosity) {
    double &valc = stencil[0];
    double &vale = stencil[1];
    double &valw = stencil[2];
    double &valn = stencil[3];
    double &vals = stencil[4];
    double &valt = stencil[5];
    double &valb = stencil[6];
    double &txc  = transform[0];
    double &txe  = transform[1];
    double &txw  = transform[2];
    double &tyc  = transform[3];
    double &tyn  = transform[4];
    double &tys  = transform[5];
    double &tzc  = transform[6];
    double &tzt  = transform[7];
    double &tzb  = transform[8];

    double diffusion = 0;
    diffusion += txc*(txc*(vale - 2*valc + valw) + 0.25*(txe - txw)*(vale - valw));
    diffusion += tyc*(tyc*(valn - 2*valc + vals) + 0.25*(tyn - tys)*(valn - vals));
    diffusion += tzc*(tzc*(valt - 2*valc + valb) + 0.25*(tzt - tzb)*(valt - valb));
    diffusion *= viscosity;
    return diffusion;
}
