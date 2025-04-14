#pragma once

#include <cstring>
#include <string>
#include <fstream>
#include "mpi_info.h"

static void build_mesh_from_directory(
    std::string path,
    double *&x,
    double *&y,
    double *&z,
    double *&dx,
    double *&dy,
    double *&dz,
    int sz[3],
    int gc,
    mpi_info *mpi
) {
    std::ifstream coord_file;
    int node_sz[3];

    coord_file.open(path + "/x.txt");
    coord_file >> node_sz[0];
    double *node_x = new double[node_sz[0]];
    for (int i = 0; i < node_sz[0]; i ++) {
        coord_file >> node_x[i];
    }
    coord_file.close();

    coord_file.open(path + "/y.txt");
    coord_file >> node_sz[1];
    double *node_y = new double[node_sz[1]];
    for (int j = 0; j < node_sz[1]; j ++) {
        coord_file >> node_y[j];
    }
    coord_file.close();

    coord_file.open(path + "/z.txt");
    coord_file >> node_sz[2];
    double *node_z = new double[node_sz[2]];
    for (int k = 0; k < node_sz[2]; k ++) {
        coord_file >> node_z[k];
    }
    coord_file.close();

    sz[0] = node_sz[0] - 1 + 2*gc;
    sz[1] = node_sz[1] - 1 + 2*gc;
    sz[2] = node_sz[2] - 1 + 2*gc;
    x  = new double[sz[0]];
    dx = new double[sz[0]];
    y  = new double[sz[1]];
    dy = new double[sz[1]];
    z  = new double[sz[2]];
    dz = new double[sz[2]];

    for (int i = gc; i < sz[0] - gc; i ++) {
        dx[i] = node_x[i - gc + 1] - node_x[i - gc];
        x[i]  = node_x[i - gc] + 0.5*dx[i];
    }
    for (int i = gc - 1; i >= 0; i --) {
        dx[i] = 2*dx[i + 1] - dx[i + 2];
        x[i]  = x[i + 1] - 0.5*(dx[i] + dx[i + 1]);
    }
    for (int i = sz[0] - gc; i < sz[0]; i ++) {
        dx[i] = 2*dx[i - 1] - dx[i - 2];
        x[i]  = x[i - 1] + 0.5*(dx[i] + dx[i - 1]);
    }

    for (int j = gc; j < sz[1] - gc; j ++) {
        dy[j] = node_y[j - gc + 1] - node_y[j - gc];
        y[j]  = node_y[j - gc] + 0.5*dy[j];
    }
    for (int j = gc - 1; j >= 0; j --) {
        dy[j] = 2*dy[j + 1] - dy[j + 2];
        y[j]  = y[j + 1] - 0.5*(dy[j] + dy[j + 1]);
    }
    for (int j = sz[1] - gc; j < sz[1]; j ++) {
        dy[j] = 2*dy[j - 1] - dy[j - 2];
        y[j]  = y[j - 1] + 0.5*(dy[j] + dy[j - 1]);
    }

    for (int k = gc; k < sz[2] - gc; k ++) {
        dz[k] = node_z[k - gc + 1] - node_z[k - gc];
        z[k]  = node_z[k - gc] + 0.5*dz[k];
    }
    for (int k = gc - 1; k >= 0; k --) {
        dz[k] = 2*dz[k + 1] - dz[k + 2];
        z[k]  = z[k + 1] - 0.5*(dz[k] + dz[k + 1]);
    }
    for (int k = sz[2] - gc; k < sz[2]; k ++) {
        dz[k] = 2*dz[k - 1] - dz[k - 2];
        z[k]  = z[k - 1] + 0.5*(dz[k] + dz[k - 1]);
    }
}

static void output_mesh(
    std::string path,
    double *x,
    double *y,
    double *z,
    double *dx,
    double *dy,
    double *dz,
    int sz[3],
    int gc
) {
    std::ofstream mesh_file(path);
    mesh_file << sz[0] << " " << sz[1] << " " << sz[2] << " " << gc << std::endl;

    for (int i = 0; i < sz[0]; i ++) {
        mesh_file << x[i] << " " << dx[i] << std::endl;
    }
    for (int j = 0; j < sz[1]; j ++) {
        mesh_file << y[j] << " " << dy[j] << std::endl;
    }
    for (int k = 0; k < sz[2]; k ++) {
        mesh_file << z[k] << " " << dz[k] << std::endl;
    }
    mesh_file.close();
}
