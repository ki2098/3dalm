#pragma once

#include <cstdint>

using Int = int64_t;
using Real = double;

template<typename T, Int N>
struct Vector {
    T m[N];

    T &operator[](Int i) {
        return m[i];
    }

    const T &operator[](Int i) const {
        return m[i];
    }
};

using Int3 = Vector<Int, 3>;
using Real3 = Vector<Real, 3>;
using Real7 = Vector<Real, 7>;
using Real9 = Vector<Real, 9>;
using Real13 = Vector<Real, 13>;