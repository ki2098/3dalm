#include <iostream>
#include <cstdlib>

using namespace std;

int main() {
    int n, m;
    cin >> n >> m;

    double **ptr = new double*[n];
    int **xx = new int*[n];
    for (int i = 0; i < n; i ++) {
        ptr[i] = new double[m];
        xx[i] = new int[m];
    }
#pragma acc enter data create(ptr[:n][:m], xx[:n][:m])
#pragma acc kernels loop independent collapse(2) present(ptr[:n][:m], xx[:n][:m])
    for (int i = 0; i < n; i ++) {
        for (int j = 0; j < m; j ++) {
            ptr[i][j] = i*3.14 + j;
            xx[i][j] = i*m + j;
        }
    }
#pragma acc exit data copyout(ptr[:n][:m], xx[:n][:m])
    for (int i = 0; i < n; i ++) {
        for (int j = 0; j < m; j ++) {
            cout << xx[i][j] << ":" << ptr[i][j] << ", ";
        }
        cout << endl;
    }

    for (int i = 0; i < n; i ++) {
        delete[] ptr[i];
        delete[] xx[i];
    }
    delete[] ptr;
    delete[] xx;
}