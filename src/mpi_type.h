#pragma once

#include <mpi.h>

template<typename T>
MPI_Datatype get_mpi_datatype() {
    if (typeid(T) == typeid(int)) return MPI_INT;
    if (typeid(T) == typeid(long)) return MPI_LONG;
    if (typeid(T) == typeid(float)) return MPI_FLOAT;
    if (typeid(T) == typeid(double)) return MPI_DOUBLE;
    return MPI_UNSIGNED_CHAR;
}