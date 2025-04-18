#pragma once

#include "type.h"

static Int getId(Int i, Int j, Int k, Int sz[3]) {
    return i*sz[1]*sz[2] + j*sz[2] + k;
}

template<typename T>
T square(T a) {
    return a*a;
}

template<typename T>
void cpyArray(T dst[], T src[], Int len) {
    #pragma acc parallel present(dst[:len], src[:len]) firstprivate(len)
    #pragma acc loop independent
    for (Int i = 0; i < len; i ++) {
        dst[i] = src[i];
    }
}

template<typename T, Int N>
void cpyArray(T dst[][N], T src[][N], Int len) {
    #pragma acc parallel present(dst[:len], src[:len]) firstprivate(len)
    #pragma acc loop independent collapse(2)
    for (Int i = 0; i < len; i ++) {
        for (Int m = 0; m < N; m ++) {
            dst[i][m] = src[i][m];
        }
    }
}

template<typename T>
void fillArray(T dst[], T value, Int len) {
    #pragma acc parallel present(dst[:len]) firstprivate(value, len)
    #pragma acc loop independent
    for (Int i = 0; i < len; i ++) {
        dst[i] = value;
    }
}

template<typename T, Int N>
void fillArray(T dst[][N], T value[N], Int len) {
    #pragma acc parallel present(dst[:len]) firstprivate(value[:N], len)
    #pragma acc loop independent collapse(2)
    for (Int i = 0; i < len; i ++) {
        for (Int m = 0; m < N; m ++) {
            dst[i][m] = value[m];
        }
    }
}