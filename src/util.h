#pragma once

#include "type.h"

Int index(Int i, Int j, Int k, Int size[3]) {
    return i*size[1]*size[2] + j*size[2] + k;
}

template<typename T>
T square(T x) {
    return x*x;
}