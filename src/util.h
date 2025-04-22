#pragma once

#include "type.h"

Int getId(Int i, Int j, Int k, Int cx, Int cy, Int cz) {
    return i*cy*cz + j*cz + k;
}

template<typename T>
T square(T x) {
    return x*x;
}

template<typename T>
void cpyArray(T *dst, T *src, Int len) {
    for (Int i = 0; i < len; i ++) {
        dst[i] = src[i];
    }
}

template<typename T>
void fillArray(T *dst, T value, Int len) {
    for (Int i = 0; i < len; i ++) {
        dst[i] = value;
    }
}