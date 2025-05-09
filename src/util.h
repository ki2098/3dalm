#pragma once

#include <string>
#include <sstream>
#include <iomanip>
#include "type.h"

#pragma acc routine seq
Int index(Int i, Int j, Int k, Int size[3]) {
    return i*size[1]*size[2] + j*size[2] + k;
}

#pragma acc routine seq
template<typename T>
T square(T x) {
    return x*x;
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
std::string to_string_fixed_length(T value, Int len, char fill_char = '0') {
    std::stringstream ss;
    ss << std::setw(len) << std::setfill(fill_char) << value;
    return ss.str();
}