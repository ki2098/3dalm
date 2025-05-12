#include <array>
#include <iostream>
#include <cmath>

using namespace std;

using int3 = array<int, 3>;
using double3 = array<double, 3>;

int index(int i, int j, int k, int3 size) {
    return i*size[1]*size[2] + j*size[2] + k;
}

double foo(double3 *u, int3 size) {
    double total = 0;
#pragma acc kernels loop independent collapse(3) \
present(u[:size[0]*size[1]*size[2]]) \
reduction(+:total)
    for (int i = 0; i < size[0]; i ++) {
    for (int j = 0; j < size[1]; j ++) {
    for (int k = 0; k < size[2]; k ++) {
        int id = index(i, j, k, size);
        total += u[id][0] + u[id][1] + u[id][2];
    }}}
    return total;
}

int main() {
    int3 size = {10, 10, 10};
    int len = size[0]*size[1]*size[2];
    double3 *u = new double3[len];

#pragma acc enter data copyin(u[:size[0]*size[1]*size[2]])

#pragma acc kernels loop independent collapse(3) \
present(u[:size[0]*size[1]*size[2]]) 
    for (int i = 0; i < size[0]; i ++) {
    for (int j = 0; j < size[1]; j ++) {
    for (int k = 0; k < size[2]; k ++) {
        int id = index(i, j, k, size);
        u[id] = {sin(id), cos(id), 0.5*(sin(id) + cos(id))};
        // u[id][0] = sin(id);
        // u[id][1] = cos(id);
        // u[id][2] = 0.5*(sin(id) + cos(id));
    }}}

    cout << foo(u, size) << endl;

#pragma acc exit data delete(u[:size[0]*size[1]*size[2]])

    delete[] u;
}