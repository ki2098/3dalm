#pragma once

#include <fstream>
#include <iostream>
#include <cstdio>
#include "type.h"
#include "util.h"

static void buildMeshFromDir(
    std::string path,
    Real *&x,
    Real *&y,
    Real *&z,
    Real *&dx,
    Real *&dy,
    Real *&dz,
    Int3 &sz,
    Int gc,
    MpiInfo *mpi
) {
    std::ifstream coordFile;
    Int3 nodeSz;

    coordFile.open(path + "/x.txt");
    coordFile >> nodeSz[0];
    double *nodeX = new double[nodeSz[0]];
    for (int i = 0; i < nodeSz[0]; i ++) {
        coordFile >> nodeX[i];
    }
    coordFile.close();

    coordFile.open(path + "/y.txt");
    coordFile >> nodeSz[1];
    double *nodeY = new double[nodeSz[1]];
    for (int j = 0; j < nodeSz[1]; j ++) {
        coordFile >> nodeY[j];
    }
    coordFile.close();

    coordFile.open(path + "/z.txt");
    coordFile >> nodeSz[2];
    double *nodeZ = new double[nodeSz[2]];
    for (int k = 0; k < nodeSz[2]; k ++) {
        coordFile >> nodeZ[k];
    }
    coordFile.close();

    sz[0] = nodeSz[0] - 1 + 2*gc;
    sz[1] = nodeSz[1] - 1 + 2*gc;
    sz[2] = nodeSz[2] - 1 + 2*gc;
    x  = new double[sz[0]];
    dx = new double[sz[0]];
    y  = new double[sz[1]];
    dy = new double[sz[1]];
    z  = new double[sz[2]];
    dz = new double[sz[2]];

    for (int i = gc; i < sz[0] - gc; i ++) {
        dx[i] = nodeX[i - gc + 1] - nodeX[i - gc];
        x[i]  = nodeX[i - gc] + 0.5*dx[i];
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
        dy[j] = nodeY[j - gc + 1] - nodeY[j - gc];
        y[j]  = nodeY[j - gc] + 0.5*dy[j];
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
        dz[k] = nodeZ[k - gc + 1] - nodeZ[k - gc];
        z[k]  = nodeZ[k - gc] + 0.5*dz[k];
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

static void writeMesh(
    std::string path,
    Real *x,
    Real *y,
    Real *z,
    Real *dx,
    Real *dy,
    Real *dz,
    Int3 sz,
    Int gc
) {
    std::ofstream meshFile(path);
    meshFile << sz[0] << " " << sz[1] << " " << sz[2] << " " << gc << std::endl;

    for (int i = 0; i < sz[0]; i ++) {
        meshFile << x[i] << " " << dx[i] << std::endl;
    }
    for (int j = 0; j < sz[1]; j ++) {
        meshFile << y[j] << " " << dy[j] << std::endl;
    }
    for (int k = 0; k < sz[2]; k ++) {
        meshFile << z[k] << " " << dz[k] << std::endl;
    }
    meshFile.close();
}

static void writeCsv(
    std::string path,
    Real *var[],
    Int varCount,
    Int *varDim,
    std::string varName[],
    Real x[],
    Real y[],
    Real z[],
    Int3 sz,
    Int gc
) {
    std::ofstream oCsv(path);

    oCsv << "x,y,z";
    for (Int n = 0; n < varCount; n ++) {
        if (varDim[n] > 1) {
            for (Int m = 0; m < varDim[n]; m ++) {
                oCsv << "," << varName[n] + std::to_string(m);
            }
        } else {
            oCsv << "," << varName[n];
        }
    }
    oCsv << std::endl;

    for (Int k = 0; k < sz[2]; k ++) {
    for (Int j = 0; j < sz[1]; j ++) {
    for (Int i = 0; i < sz[0]; i ++) {
        oCsv << x[i] << "," << y[j] << "," << z[k];
        for (Int n = 0; n < varCount; n ++) {
            for (Int m = 0; m < varDim[n]; m ++) {
                Int id = getId(i, j, k, sz)*varDim[n] + m;
                oCsv << "," << var[n][id];
            }
        }
        oCsv << std::endl;
    }}}

    oCsv.close();
}
