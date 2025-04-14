#pragma once

static int getid(int i, int j, int k, int sz[3]) {
    return i*sz[1]*sz[2] + j*sz[2] + k;
}

template<typename T>
static T square(T a) {
    return a*a;
}
