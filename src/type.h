#pragma once

#include <cstdint>

using Int = int64_t;
using Real = double;

struct MpiInfo {
    int size, rank;
};