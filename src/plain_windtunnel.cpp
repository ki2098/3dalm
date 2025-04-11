#include <iostream>
#include "cfd.h"
#include "io.h"
#include "mpi_info.h"
#include "pbicgstab.h"
#include "boundary_condition.h"
#include "mesh.h"
#include "mv.h"
#include "json.hpp"

#define GUIDE_CELL 2

using namespace std;

using json = nlohmann::json;

struct runtime_t {
    double dt;
    int max_step;
    int step;

    void init(double dt, int max_step) {
        cout << "RUNTIME INIT" << endl;
        this->dt = dt;
        this->max_step = max_step;
        cout << "\tdt " << this->dt << endl;
        cout << "\tmax step " << this->max_step << endl;
    }
};

struct mesh_t {
    double *x;
    double *y;
    double *z;
    double *dx;
    double *dy;
    double *dz;

    void init(string path, int sz[3], int gc, mpi_info *mpi) {
        cout << "MESH INIT" << endl;
        cout << "\treading mesh from " << path << "...";
        build_mesh_from_directory(path, x, y, z, dx, dy, dz, sz, gc, mpi);

        #pragma acc enter data \
        copyin(x[:sz[0]], y[:sz[1]], z[:sz[2]]) \
        copyin(dx[:sz[0]], dy[:sz[1]], dz[:sz[2]])

        cout << "done" << endl;
        cout << "\ttotal cells " << sz[0] << "x" << sz[1] << "x" << sz[2] << endl;
        cout << "\tguid cells " << gc << endl;
    }

    void finalize(int sz[3]) {
        delete[] x;
        delete[] y;
        delete[] z;
        delete[] dx;
        delete[] dy;
        delete[] dz;

        #pragma acc exit data \
        delete(x[:sz[0]], y[:sz[1]], z[:sz[2]]) \
        delete(dx[:sz[0]], dy[:sz[1]], dz[:sz[2]])
    }
};

struct cfd_t {
    double (*U)[3];
    double (*Uprev)[3];
    double (*Utavg)[3];
    double *p;
    double *nut;
    double Re;
    double Cs;
    double divergence_monitor;

    void init(double Re, double Cs, int sz[3]) {
        cout << "CFD INIT" << endl;
        this->Re = Re;
        this->Cs = Cs;
        int cnt = sz[0]*sz[1]*sz[2];
        U = new double[cnt][3];
        Uprev = new double[cnt][3];
        Utavg = new double[cnt][3];
        p = new double[cnt];
        nut = new double[cnt];

        #pragma acc enter data \
        create(U[:cnt], Uprev[:cnt], Utavg[:cnt], p[:cnt], nut[:cnt])

        cout << "\tRe " << this->Re << endl;
        cout << "\tCs " << this->Cs << endl;
    }

    void finalize(int sz[3]) {
        delete[] U;
        delete[] Uprev;
        delete[] Utavg;
        delete[] p;
        delete[] nut;
        int cnt = sz[0]*sz[1]*sz[2];
        #pragma acc exit data \
        delete(U[:cnt], Uprev[:cnt], Utavg[:cnt], p[:cnt], nut[:cnt])
    }
};

struct ls_t {
    double (*A)[7];
    double *b;
    double *r;
    double *r0;
    double *p;
    double *q;
    double *s;
    double *phat;
    double *shat;
    double *t;
    double *tmp;
    double err;
    double tolerance;
    int iteration;
    int max_iteration;

    void init(double tolerance, int max_iteration, int sz[3]) {
        cout << "LS INIT" << endl;
        this->tolerance = tolerance;
        this->max_iteration = max_iteration;
        int cnt = sz[0]*sz[1]*sz[2];
        A = new double[cnt][7];
        b = new double[cnt];
        r = new double[cnt];
        r0 = new double[cnt];
        p = new double[cnt];
        q = new double[cnt];
        s = new double[cnt];
        phat = new double[cnt];
        shat = new double[cnt];
        t = new double[cnt];
        tmp = new double[cnt];

        #pragma acc enter data \
        create(A[:cnt], b[:cnt], r[:cnt]) \
        create(r0[:cnt], p[:cnt], q[:cnt], s[:cnt], phat[:cnt], shat[:cnt], t[:cnt], tmp[:cnt])

        cout << "\ttolerance " << this->tolerance << endl;
        cout << "\tmax iteration " << this->max_iteration << endl;
    }

    void finalize(int sz[3]) {
        delete[] A;
        delete[] b;
        delete[] r;
        delete[] r0;
        delete[] p;
        delete[] q;
        delete[] s;
        delete[] phat;
        delete[] shat;
        delete[] t;
        delete[] tmp;
        int cnt = sz[0]*sz[1]*sz[2];
        #pragma acc exit data \
        delete(A[:cnt], b[:cnt], r[:cnt]) \
        delete(r0[:cnt], p[:cnt], q[:cnt], s[:cnt], phat[:cnt], shat[:cnt], t[:cnt], tmp[:cnt])
    }
};

void init(string setup_path, runtime_t *runtime, mesh_t *mesh, cfd_t *cfd, ls_t *ls, mpi_info *mpi, int sz[3]) {
    ifstream setup_file(setup_path);
    auto setup_json = json::parse(setup_file);

    auto &rt_json = setup_json["runtime"];
    double dt = rt_json["dt"];
    double total_time = rt_json["time"];
    runtime->init(dt, total_time/dt);

    auto &mesh_json = setup_json["mesh"];
    string mesh_path = mesh_json["path"];
    mesh->init(mesh_path, sz, GUIDE_CELL, mpi);

    auto &cfd_json = setup_json["cfd"];
    double Re = cfd_json["Re"];
    double Cs = cfd_json["Cs"];
    cfd->init(Re, Cs, sz);

    auto &ls_json = setup_json["ls"];
    double tolerance = ls_json["tolerance"];
    int max_iteration = ls_json["max_iteration"];
    ls->init(tolerance, max_iteration, sz);
}

void finalize(mesh_t *mesh, cfd_t *cfd, ls_t *ls, int sz[3]) {
    mesh->finalize(sz);
    cfd->finalize(sz);
    ls->finalize(sz);
}

int main(int argc, char *argv[]) {
    string tom_path(argv[1]);
    int sz[3];
    runtime_t runtime;
    mesh_t mesh;
    cfd_t cfd;
    ls_t ls;
    mpi_info mpi;

    init(tom_path, &runtime, &mesh, &cfd, &ls, &mpi, sz);

    finalize(&mesh, &cfd, &ls, sz);
}