#pragma once

#include "type.h"

static Int getId(Int i, Int j, Int k, Int3 sz) {
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

template<typename T>
void fillArray(T dst[], T value, Int len) {
    for (Int i = 0; i < len; i ++) {
        
    }
}