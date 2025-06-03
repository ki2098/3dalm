#include <mpi.h>
#include <cstdio>
#include <cstdlib>

struct MyStruct {
    int a;
    double x[3];
};

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MyStruct array[100];
    int trunk = 100/size;
    if (rank < 100%size) {
        trunk ++;
    }
    int offset = 0;
    for (int rank_ = 0; rank_ < rank; rank_ ++) {
        int trunk_ = 100/size;
        if (rank_ < 100%size) {
            trunk_ ++;
        }
        offset += trunk_;
    }
    printf("%d %d %d\n", rank, offset, trunk);
#pragma acc enter data create(array[:100])
#pragma acc kernels loop independent present(array[:100])
    for (int i = 0; i < 100; i ++) {
        if (i >= offset && i < offset + trunk) {
            array[i].a = i;
            array[i].x[0] = i*3.14;
            array[i].x[1] = i*3.14 + 1;
            array[i].x[2] = i*3.14 + 2;
        } else {
            array[i].a = 0;
            array[i].x[0] = 0;
            array[i].x[1] = 0;
            array[i].x[2] = 0;
        }
    }

    for (int i = 0; i < 100; i ++) {
#pragma acc host_data use_device(array)
        MPI_Allreduce(MPI_IN_PLACE, &array[i].a, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#pragma acc host_data use_device(array)
        MPI_Allreduce(MPI_IN_PLACE, array[i].x, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }

#pragma acc exit data copyout(array[:100])
    if (rank == 0) {
        for (int i = 0; i < 100; i ++) {
            printf("%d %lf %lf %lf\n", array[i].a, array[i].x[0], array[i].x[1], array[i].x[2]);
        }
    }

    MPI_Finalize();
}