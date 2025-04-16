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
    for (Int i = 0; i < len; i ++) {
        dst[i] = src[i];
    }
}

template<typename T, Int N>
void cpyArray(T dst[][N], T src[][N], Int len) {
    for (Int i = 0; i < len; i ++) {
        for (Int m = 0; m < N; m ++) {
            dst[i][m] = src[i][m];
        }
    }
}