#pragma once

#include <string>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include "type.h"

#pragma acc routine seq
static Int index(Int i, Int j, Int k, Int size[3]) {
    return i*size[1]*size[2] + j*size[2] + k;
}

#pragma acc routine seq
template<typename T>
T square(T x) {
    return x*x;
}

#pragma acc routine seq
template<typename T>
T cubic(T x) {
    return x*x*x;
}

template<typename T>
void cpy_array(T dst[], T src[], Int len) {
#pragma acc kernels loop independent \
present(dst[:len], src[:len])
    for (Int i = 0; i < len; i ++) {
        dst[i] = src[i];
    }
}

template<typename T, Int N>
void cpy_array(T dst[][N], T src[][N], Int len) {
#pragma acc kernels loop independent \
present(dst[:len], src[:len])
    for (Int i = 0; i < len; i ++) {
#pragma acc loop seq
        for (Int m = 0; m < N; m ++) {
            dst[i][m] = src[i][m];
        }
    }
}

template<typename T>
void fill_array(T dst[], T value, Int len) {
#pragma acc kernels loop independent \
present(dst[:len])
    for (Int i = 0; i < len; i ++) {
        dst[i] = value;
    }
}

template<typename T, Int N>
void fill_array(T dst[][N], T value[N], Int len) {
#pragma acc kernels loop independent \
present(dst[:len]) \
copyin(value[:N])
    for (Int i = 0; i < len; i ++) {
#pragma acc loop seq
        for (Int m = 0; m < N; m ++) {
            dst[i][m] = value[m];
        }
    }
}

#pragma acc routine seq
template<typename T>
Int sign(T x) {
    return (x > T(0)) - (x < T(0));
}

template<typename T>
std::string to_str_fixed_length(T value, Int len, char fill_char = '0') {
    std::stringstream ss;
    ss << std::setw(len) << std::setfill(fill_char) << value;
    return ss.str();
}

#pragma acc routine seq
static Real get_intersection(Real h1, Real t1, Real h2, Real t2) {
    Real l1 = t1 - h1;
    Real l2 = t2 - h2;
    Real x1 = t1 - h2;
    Real x2 = t2 - h1;
    return fmax(0., fmin(l1, fmin(l2, fmin(x1, x2))));
}

template<typename T>
Int find_nearest_index(T *arr, T value, Int len) {
    if (value < arr[0]) {
        return 0;
    }
    if (value >= arr[len - 1]) {
        return len - 1;
    }
    for (Int i = 0; i < len - 1; i ++) {
        if (arr[i] <= value && value < arr[i + 1]) {
            return (value - arr[i] < arr[i + 1] - value)? i : i + 1;
        }
    }
    return - 1;
}

#pragma acc routine seq
template<typename T>
Int find_floor_index(T *arr, T value, Int len) {
    if (value < arr[0]) {
        return - 1;
    }
    if (value >= arr[len - 1]) {
        return len - 1;
    }
    for (Int i = 0; i < len - 1; i ++) {
        if (arr[i] <= value && value < arr[i + 1]) {
            return i;
        }
    }
    return - 1;
}

#pragma acc routine seq
static Int euclid_mod(Int x, Int y) {
    Int r = x%y;
    if (r < 0) {
        r += abs(y);
    }
    return r;
}

#pragma acc routine seq
static Real euclid_fmod(Real x, Real y) {
    Real r = fmod(x, y);
    if (r < 0) {
        r += fabs(y);
    }
    return r;
}

#pragma acc routine seq
static Real linear_interpolate(
    Real v0, Real v1,
    Real x0, Real x1,
    Real xp
) {
    Real fx = (xp - x0)/(x1 - x0);
    return v0*(1 - fx) + v1*fx;
}

#pragma acc routine seq
static Real trilinear_interpolate(
    Real v0, Real v1, Real v2, Real v3, Real v4, Real v5, Real v6, Real v7,
    Real x0, Real x1,
    Real y0, Real y1,
    Real z0, Real z1,
    Real xp, Real yp, Real zp
) {
    Real fx = (xp - x0)/(x1 - x0);
    Real fy = (yp - y0)/(y1 - y0);
    Real fz = (zp - z0)/(z1 - z0);

    Real v01 = v0*(1 - fx) + v1*fx;
    Real v23 = v2*(1 - fx) + v3*fx;
    Real v45 = v4*(1 - fx) + v5*fx;
    Real v67 = v6*(1 - fx) + v7*fx;

    Real v0123 = v01*(1 - fy) + v23*fy;
    Real v4567 = v45*(1 - fy) + v67*fy;

    Real vp = v0123*(1 - fz) + v4567*fz;

    return vp;
}

#pragma acc routine seq
static Real quadratic_polynomial(
    Real x0, Real x1, Real x2,
    Real f0, Real f1, Real f2,
    Real x3
) {
    Real l0 = ((x3 - x1) * (x3 - x2)) / ((x0 - x1) * (x0 - x2));
    Real l1 = ((x3 - x0) * (x3 - x2)) / ((x1 - x0) * (x1 - x2));
    Real l2 = ((x3 - x0) * (x3 - x1)) / ((x2 - x0) * (x2 - x1));

    return f0*l0 + f1*l1 + f2*l2;
}

static Real degree_to_rad(Real a) {
    return a*Pi/180;
}