#pragma once

#include <fstream>
#include <iostream>
#include <cstdio>
#include "util.h"

static void build_mesh(
    std::string path,
    Real *&gx, Real *&gy, Real *&gz,
    Real *&gdx, Real *&gdy, Real *&gdz,
    Real *&x, Real *&y, Real *&z,
    Real *&dx, Real *&dy, Real *&dz,
    Int gsize[3], Int size[3], Int offset[3], Int gc,
    MpiInfo *mpi
) {
    
}

static void build_mesh(
    std::string path,
    Real *&x, Real *&y, Real *&z,
    Real *&dx, Real *&dy, Real *&dz,
    Int size[3], Int gc,
    MpiInfo *mpi
) {
    std::ifstream coord_file;
    Int node_size[3];

    coord_file.open(path + "/x.txt");
    coord_file >> node_size[0];
    double *node_x = new double[node_size[0]];
    for (int i = 0; i < node_size[0]; i ++) {
        coord_file >> node_x[i];
    }
    coord_file.close();

    coord_file.open(path + "/y.txt");
    coord_file >> node_size[1];
    double *node_y = new double[node_size[1]];
    for (int j = 0; j < node_size[1]; j ++) {
        coord_file >> node_y[j];
    }
    coord_file.close();

    coord_file.open(path + "/z.txt");
    coord_file >> node_size[2];
    double *node_z = new double[node_size[2]];
    for (int k = 0; k < node_size[2]; k ++) {
        coord_file >> node_z[k];
    }
    coord_file.close();

    size[0] = node_size[0] - 1 + 2*gc;
    size[1] = node_size[1] - 1 + 2*gc;
    size[2] = node_size[2] - 1 + 2*gc;
    x  = new double[size[0]];
    dx = new double[size[0]];
    y  = new double[size[1]];
    dy = new double[size[1]];
    z  = new double[size[2]];
    dz = new double[size[2]];

    for (int i = gc; i < size[0] - gc; i ++) {
        dx[i] = node_x[i - gc + 1] - node_x[i - gc];
        x[i]  = node_x[i - gc] + 0.5*dx[i];
    }
    for (int i = gc - 1; i >= 0; i --) {
        dx[i] = 2*dx[i + 1] - dx[i + 2];
        x[i]  = x[i + 1] - 0.5*(dx[i] + dx[i + 1]);
    }
    for (int i = size[0] - gc; i < size[0]; i ++) {
        dx[i] = 2*dx[i - 1] - dx[i - 2];
        x[i]  = x[i - 1] + 0.5*(dx[i] + dx[i - 1]);
    }

    for (int j = gc; j < size[1] - gc; j ++) {
        dy[j] = node_y[j - gc + 1] - node_y[j - gc];
        y[j]  = node_y[j - gc] + 0.5*dy[j];
    }
    for (int j = gc - 1; j >= 0; j --) {
        dy[j] = 2*dy[j + 1] - dy[j + 2];
        y[j]  = y[j + 1] - 0.5*(dy[j] + dy[j + 1]);
    }
    for (int j = size[1] - gc; j < size[1]; j ++) {
        dy[j] = 2*dy[j - 1] - dy[j - 2];
        y[j]  = y[j - 1] + 0.5*(dy[j] + dy[j - 1]);
    }

    for (int k = gc; k < size[2] - gc; k ++) {
        dz[k] = node_z[k - gc + 1] - node_z[k - gc];
        z[k]  = node_z[k - gc] + 0.5*dz[k];
    }
    for (int k = gc - 1; k >= 0; k --) {
        dz[k] = 2*dz[k + 1] - dz[k + 2];
        z[k]  = z[k + 1] - 0.5*(dz[k] + dz[k + 1]);
    }
    for (int k = size[2] - gc; k < size[2]; k ++) {
        dz[k] = 2*dz[k - 1] - dz[k - 2];
        z[k]  = z[k - 1] + 0.5*(dz[k] + dz[k - 1]);
    }

    delete[] node_x;
    delete[] node_y;
    delete[] node_z;
}

static void write_mesh(
    std::string path,
    Real x[], Real y[], Real z[],
    Real dx[], Real dy[], Real dz[],
    Int size[3], Int gc
) {
    std::ofstream mesh_file(path);
    mesh_file << size[0] << " " << size[1] << " " << size[2] << " " << gc << std::endl;

    for (int i = 0; i < size[0]; i ++) {
        mesh_file << x[i] << " " << dx[i] << std::endl;
    }
    for (int j = 0; j < size[1]; j ++) {
        mesh_file << y[j] << " " << dy[j] << std::endl;
    }
    for (int k = 0; k < size[2]; k ++) {
        mesh_file << z[k] << " " << dz[k] << std::endl;
    }
    mesh_file.close();
}

static void write_csv(
    std::string path,
    Real *var[], Int var_count, Int var_dim[], std::string var_name[],
    Real x[], Real y[], Real z[],
    Int size[3], Int gc
) {
    std::ofstream ocsv(path);

    ocsv << "x,y,z";
    for (Int v = 0; v < var_count; v ++) {
        if (var_dim[v] > 1) {
            for (Int m = 0; m < var_dim[v]; m ++) {
                ocsv << "," << var_name[v] + std::to_string(m);
            }
        } else {
            ocsv << "," << var_name[v];
        }
    }
    ocsv << std::endl;

    for (Int k = 0; k < size[2]; k ++) {
    for (Int j = 0; j < size[1]; j ++) {
    for (Int i = 0; i < size[0]; i ++) {
        ocsv << x[i] << "," << y[j] << "," << z[k];
        for (Int v = 0; v < var_count; v ++) {
            for (Int m = 0; m < var_dim[v]; m ++) {
                Int id = index(i, j, k, size)*var_dim[v] + m;
                ocsv << "," << var[v][id];
            }
        }
        ocsv << std::endl;
    }}}
}

static void write_binary(
    std::string path,
    Real *var[], Int var_count, Int *var_dim, std::string var_name[],
    Real x[], Real y[], Real z[],
    Int size[3], Int gc
) {
    std::ofstream ofs(path, std::ios::binary);
    ofs.write((char*)size, 3*sizeof(Int));
    ofs.write((char*)&gc, sizeof(Int));
    ofs.write((char*)&var_count, sizeof(Int));
    ofs.write((char*)var_dim, var_count*sizeof(Int));
    for (Int v = 0; v < var_count; v ++) {
        auto &s = var_name[v];
        Int len = s.length();
        ofs.write((char*)&len, sizeof(Int));
        ofs.write((char*)s.c_str(), sizeof(char)*len);
    }
    ofs.write((char*)x, size[0]*sizeof(Real));
    ofs.write((char*)y, size[1]*sizeof(Real));
    ofs.write((char*)z, size[2]*sizeof(Real));
    for (Int v = 0; v < var_count; v ++) {
        Int count = size[0]*size[1]*size[2]*var_dim[v];
        ofs.write((char*)var[v], count*sizeof(Real));
    }
    ofs.close();
}