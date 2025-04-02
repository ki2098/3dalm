#include <cstdio>
#include <iostream>

/* template<typename T, int N>
struct vector_t {
    T a_[N];

    T &operator[](int i) {
        return a_[i];
    }
}; */

using int3_t = int[3];

inline int getid(int i, int j, int k, int size[3]) {
    return i*size[1]*size[2] + j*size[2] + k;
}

void foo(
    int data[][3],
    int sz[3]
) {
    #pragma acc parallel loop independent collapse(3) \
    present(data[:sz[0]*sz[1]*sz[2]]) \
    firstprivate(sz[:3])
    for (int i = 0; i < sz[0]; i ++) {
        for (int j = 0; j < sz[1]; j ++) {
            for (int k = 0; k < sz[2]; k ++) {
                data[getid(i, j, k, sz)][0] = getid(i, j, k,sz)*3;
                data[getid(i, j, k, sz)][1] = getid(i, j, k,sz)*3 + 1;
                data[getid(i, j, k, sz)][2] = getid(i, j, k,sz)*3 + 2;
                printf("%d %d %d\n", data[getid(i, j, k, sz)][0], data[getid(i, j, k, sz)][1], data[getid(i, j, k, sz)][2]);
            }
        }
    }
}

int main() {
    int sz[] = {5, 5, 5};
    int cnt = sz[0]*sz[1]*sz[2];
    int (*data)[3] = new int[cnt][3];

    #pragma acc enter data create(data[:sz[0]*sz[1]*sz[2]])

    foo(data, sz);

    #pragma acc exit data delete(data[:sz[0]*sz[1]*sz[2]])

    delete[] data;
}