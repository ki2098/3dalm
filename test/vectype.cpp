#include <cmath>
#include <cstdio>
#include <iostream>
#include <cstdint>
#include <chrono>

using namespace std;

using Real = double;
using Int = int64_t;

Int index(Int i, Int j, Int k, Int *sz) {
    return i*sz[1]*sz[2] + j*sz[2] + k;
}

Real scheme(Real *stencil) {
    Real valcc = stencil[0];
    Real vale1 = stencil[1];
    Real vale2 = stencil[2];
    Real valw1 = stencil[3];
    Real valw2 = stencil[4];
    return vale2 - 4*vale1 + 6*valcc - 4*valw1 + valw2;
}

void foo(Real (*data)[3], Int *sz, Int gc) {
    Int len = sz[0]*sz[1]*sz[2];
#pragma acc kernels loop independent collapse(3) present(data[:len])
    for (Int i = gc; i < sz[0] - gc; i ++) {
    for (Int j = gc; j < sz[1] - gc; j ++) {
    for (Int k = gc; k < sz[2] - gc; k ++) {
        Int id = index(i, j, k, sz);
        data[id][0] = sin(id);
        data[id][1] = cos(id);
        data[id][2] = 0.5*(sin(id) + cos(id));
    }}}
}

void bar(Real (*data)[3], Real (*result)[3], Real scale, Int *sz, Int gc) {
    Int len = sz[0]*sz[1]*sz[2];
#pragma acc kernels loop independent collapse(3) present(data[:len], result[:len])
    for (Int i = gc; i < sz[0] - gc; i ++) {
    for (Int j = gc; j < sz[1] - gc; j ++) {
    for (Int k = gc; k < sz[2] - gc; k ++) {
        Int idcc = index(i, j, k, sz);
        Int ide1 = index(i + 1, j, k, sz);
        Int ide2 = index(i + 2, j, k, sz);
        Int idw1 = index(i - 1, j, k, sz);
        Int idw2 = index(i - 2, j, k, sz);
        for (Int m = 0; m < 3; m ++) {
            Real stencil[] = {data[idcc][m], data[ide1][m], data[ide2][m], data[idw1][m], data[idw2][m]};
            result[idcc][m] = scheme(stencil)/scale;
        }
    }}}
}

int main() {
    Int gc = 2;
    Int sz[] = {1000 + 2*gc, 300 + 2*gc, 300 + 2*gc};
    Int len = sz[0]*sz[1]*sz[2];

    Real (*data)[3] = new Real[len][3];
    Real (*result)[3] = new Real[len][3];

#pragma acc enter data create(data[:len], result[:len])

    auto start = chrono::high_resolution_clock::now();

    foo(data, sz, gc);
    bar(data, result, 10, sz, gc);
    
    Real total = 0;
#pragma acc kernels loop independent collapse(3) present(result[:len]) reduction(+:total)
    for (Int i = gc; i < sz[0] - gc; i ++) {
    for (Int j = gc; j < sz[1] - gc; j ++) {
    for (Int k = gc; k < sz[2] - gc; k ++) {
        Int id = index(i, j, k, sz);
        total += result[id][0] + result[id][1] + result[id][2];
    }}}

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << total << " " << duration.count() << endl;

#pragma acc exit data delete(data[:len], result[:len])

    delete[] data;
    delete[] result;
}