#include <cstdio>
#include <cstdint>
#include <cmath>
#include <chrono>
// #include "../src/type.h"
// #include "../src/util.h"

using Int = int64_t;
using Real = double;

using namespace std;

template<typename T, int N>
struct Vector {
    T m[N];

    T &operator[](int i) {
        return m[i];
    }
    
    const T &operator[](int i) const {
        return m[i];
    }
};

using Int3 = Vector<Int, 3>;
using Real3 = Vector<Real, 3>;

static Int getId(Int i, Int j, Int k, Int3 sz) {
    return i*sz[1]*sz[2] + j*sz[2] + k;
}

void foo(Real3 *data, Int3 sz, Int gc) {
    Int len = sz[0]*sz[1]*sz[2];
#pragma acc parallel loop independent collapse(3) present(data[:len]) copyin(sz, sz.m[:3])
    for (Int i = gc; i < sz[0] - gc; i ++) {
    for (Int j = gc; j < sz[1] - gc; j ++) {
    for (Int k = gc; k < sz[2] - gc; k ++) {
        Int id = getId(i, j, k, sz);
        data[id][0] = sin(id);
        data[id][1] = cos(id);
        data[id][2] = 0.5*(sin(id) + cos(id));
    }}}
}

int main() {
    Int gc = 2;
    Int3 sz = {1000 + 2*gc, 300 + 2*gc, 300 + 2*gc};
    Int len = sz[0]*sz[1]*sz[2];

    Real3 *data = new Real3[len];

#pragma acc enter data create(data[:len])

    auto start = chrono::high_resolution_clock::now();
    foo(data, sz, gc);
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);

    printf("%ld\n", duration.count());

#pragma acc update host(data[:len])

    Real total = 0;
    for (Int i = gc; i < sz[2] - gc; i ++) {
    for (Int j = gc; j < sz[1] - gc; j ++) {
    for (Int k = gc; k < sz[2] - gc; k ++) {
        Int id = getId(i, j, k, sz);
        total += data[id][0] + data[id][1] + data[id][2];
    }}}
    printf("%lf\n", total);

    delete[] data;

#pragma acc exit data delete(data[:len])
}