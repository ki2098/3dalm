#pragma once

#include <cstdint>
#include <string>

using Int = int64_t;
using Real = double;

template<typename T, Int N>
using Vector = T[N];

using Int3 = Vector<Int, 3>;
using Real3 = Vector<Real, 3>;
using Real7 = Vector<Real, 7>;
using Real9 = Vector<Real, 9>;
using Real13 = Vector<Real, 13>;

struct MpiInfo {
    int rank;
    int size;
};