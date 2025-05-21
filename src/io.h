#pragma once

#include <fstream>
#include <iostream>
#include <cstdio>
#include <vector>
#include <cassert>
#include "util.h"
#include "json.hpp"

struct Header {
    Int size[3];
    Int gc;
    Int var_count;
    std::vector<Int> var_dim;
    std::vector<std::string> var_name;

    void print_info() {
        printf("size = (%ld %ld %ld)\n", size[0], size[1], size[2]);
        printf("gc = %ld\n", gc);
        printf("var count = %ld\n", var_count);
        printf("var dim = (\n");
        for (int v = 0; v < var_count; v ++) {
            printf("\t%ld\n", var_dim[v]);
        }
        printf(")\n");
        printf("var name = (\n");
        for (auto &s : var_name) {
            printf("\t%s\n", s.c_str());
        }
        printf(")\n");
    }

    void write(std::ofstream &ofs) {
        ofs.write((char*)size, 3*sizeof(Int));
        ofs.write((char*)&gc, sizeof(Int));
        ofs.write((char*)&var_count, sizeof(Int));
        ofs.write((char*)var_dim.data(), var_count*sizeof(Int));
        for (Int v = 0; v < var_count; v ++) {
            auto &s = var_name[v];
            Int len = s.length();
            ofs.write((char*)&len, sizeof(Int));
            ofs.write((char*)s.c_str(), sizeof(char)*len);
        }
    }

    void read(std::ifstream &ifs) {
        ifs.read((char*)size, 3*sizeof(Int));
        ifs.read((char*)&gc, sizeof(Int));
        ifs.read((char*)&var_count, sizeof(Int));
        var_dim.resize(var_count);
        ifs.read((char*)var_dim.data(), var_count*sizeof(Int));
        var_name.resize(var_count);
        for (Int v = 0; v < var_count; v ++) {
            Int len;
            ifs.read((char*)&len, sizeof(Int));
            std::string s(len, '\0');
            ifs.read((char*)s.c_str(), len*sizeof(char));
            var_name[v] = s;
        }
    }
};

struct OutHandler : public Header {
    std::vector<Real*> var;

    void set_size(Int size[3], Int gc) {
        this->size[0] = size[0];
        this->size[1] = size[1];
        this->size[2] = size[2];
        this->gc = gc;
    }

    void set_var(
        const std::vector<Real*> &var,
        const std::vector<Int> &var_dim,
        const std::vector<std::string> &var_name
    ) {
        this->var_count = var.size();
        this->var = var;
        this->var_dim = var_dim;
        this->var_name = var_name;
        assert(
            var_dim.size() == var_count &&
            var_name.size() == var_count
        );
    }

    void update_host() {
        for (Int v = 0; v < var_count; v ++) {
            Real *ptr = var[v];
            Int len = size[0]*size[1]*size[2]*var_dim[v];
#pragma acc update host(ptr[:len])
        }
    }
};

static void build_mesh(
    const std::string &path,
    Real *&x, Real *&y, Real *&z,
    Real *&dx, Real *&dy, Real *&dz,
    Int size[3], Int &gc,
    MpiInfo *mpi
) {
    // std::ifstream x_coord_file(directory + "/x.txt");
    // x_coord_file >> node_size[0];
    // double *node_x = new double[node_size[0]];
    // for (int i = 0; i < node_size[0]; i ++) {
    //     x_coord_file >> node_x[i];
    // }

    // std::ifstream y_coord_file(directory + "/y.txt");
    // y_coord_file >> node_size[1];
    // double *node_y = new double[node_size[1]];
    // for (int j = 0; j < node_size[1]; j ++) {
    //     y_coord_file >> node_y[j];
    // }

    // std::ifstream z_coord_file(directory + "/z.txt");
    // z_coord_file >> node_size[2];
    // double *node_z = new double[node_size[2]];
    // for (int k = 0; k < node_size[2]; k ++) {
    //     z_coord_file >> node_z[k];
    // }
    std::ifstream coord_file(path);
    coord_file >> size[0] >> size[1] >> size[2] >> gc;
    x  = new double[size[0]];
    dx = new double[size[0]];
    y  = new double[size[1]];
    dy = new double[size[1]];
    z  = new double[size[2]];
    dz = new double[size[2]];
    for (Int i = 0; i < size[0]; i ++) {
        coord_file >> x[i] >> dx[i];
    }
    for (Int j = 0; j < size[1]; j ++) {
        coord_file >> y[j] >> dy[j];
    }
    for (Int k = 0; k < size[2]; k ++) {
        coord_file >> z[k] >> dz[k];
    }
}

static void write_mesh(
    const std::string &path,
    Real *x, Real *y, Real *z,
    Real *dx, Real *dy, Real *dz,
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
}

static void write_csv(
    const std::string &path,
    OutHandler *handler, Real x[], Real y[], Real z[]
) {
    std::ofstream ocsv(path);

    Int var_count = handler->var_count;
    auto &var_dim = handler->var_dim;
    auto &var_name = handler->var_name;
    auto &var = handler->var;
    auto size = handler->size;

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
    const std::string &path,
    OutHandler *handler, Real *x, Real *y, Real *z
) {
    auto size = handler->size;
    std::ofstream ofs(path, std::ios::binary);
    handler->write(ofs);
    ofs.write((char*)x, size[0]*sizeof(Real));
    ofs.write((char*)y, size[1]*sizeof(Real));
    ofs.write((char*)z, size[2]*sizeof(Real));
    Int var_count = handler->var_count;
    auto &var_dim = handler->var_dim;
    auto &var = handler->var;
    for (Int v = 0; v < var_count; v ++) {
        Int count = size[0]*size[1]*size[2]*var_dim[v];
        ofs.write((char*)var[v], count*sizeof(Real));
    }
}

static std::string make_rank_binary_filename(std::string prefix, int rank, Int step) {
    return prefix + "_" + to_string_fixed_length(rank, 5) + "_" + to_string_fixed_length(step, 10);
}

static std::string make_binary_filename(std::string prefix, Int step) {
    return prefix + "_" + to_string_fixed_length(step, 10);
}