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
    double Uin[3];
    double *p;
    double *nut;
    double Re;
    double Cs;
    double divergence_monitor;

    void init(double Uin[3], double Re, double Cs, int sz[3]) {
        cout << "CFD INIT" << endl;
        this->Re = Re;
        this->Cs = Cs;
        this->Uin[0] = Uin[0];
        this->Uin[1] = Uin[1];
        this->Uin[2] = Uin[2];
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
        cout << "\tUin (" << this->Uin[0] << " " << this->Uin[1] << " " << this->Uin[2] << ")" << endl;
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
    double max_diag;

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

    int cnt = sz[0]*sz[1]*sz[2];

    auto &cfd_json = setup_json["cfd"];
    auto &inflow_json = setup_json["inflow"];
    double Re = cfd_json["Re"];
    double Cs = cfd_json["Cs"];
    double Uin[3];
    Uin[0] = inflow_json["value"][0];
    Uin[1] = inflow_json["value"][1];
    Uin[2] = inflow_json["value"][2];
    cfd->init(Uin, Re, Cs, sz);

    fill_array(cfd->U, cfd->Uin, cnt);
    fill_array(cfd->p, 0, cnt);
    calc_eddy_viscosity(cfd->U, cfd->nut, cfd->Cs, mesh->dx, mesh->dy, mesh->dz, sz, GUIDE_CELL, mpi);
    cfd->divergence_monitor = monitor_divergence(cfd->U, mesh->dx, mesh->dy, mesh->dz, sz, GUIDE_CELL, mpi);
    cout << "initial divergence " << cfd->divergence_monitor << endl;

    auto &ls_json = setup_json["ls"];
    double tolerance = ls_json["tolerance"];
    int max_iteration = ls_json["max_iteration"];
    ls->init(tolerance, max_iteration, sz);

    ls->max_diag = build_coefficient_matrix(ls->A, mesh->dx, mesh->dy, mesh->dz, sz, GUIDE_CELL, mpi);
}

void main_loop(runtime_t *runtime, mesh_t *mesh, cfd_t *cfd, ls_t *ls, mpi_info *mpi, int sz[3]) {
    int cnt = sz[0]*sz[1]*sz[2];
    cpy_array(cfd->Uprev, cfd->U, cnt);

    calc_pseudo_velocity(cfd->U, cfd->Uprev, cfd->nut, cfd->Re, runtime->dt, mesh->dx, mesh->dy, mesh->dz, sz, GUIDE_CELL, mpi);

    calc_poisson_rhs(cfd->U, ls->b, runtime->dt, ls->max_iteration, mesh->dx, mesh->dy, mesh->dz, sz, GUIDE_CELL, mpi);

    run_pbicgstab(ls->A, cfd->p, ls->b, ls->r, ls->r0, ls->p, ls->q, ls->s, ls->phat, ls->shat, ls->t, ls->tmp, ls->err, ls->tolerance, ls->iteration, ls->max_iteration, sz, GUIDE_CELL, mpi);

    apply_p_boundary_condition(cfd->p, cfd->Uprev, cfd->Re, cfd->nut, mesh->z, mesh->dz, sz, GUIDE_CELL, mpi);

    project_pressure(cfd->p, cfd->U, runtime->dt, mesh->dx, mesh->dy, mesh->dz, sz, GUIDE_CELL, mpi);

    apply_U_boundary_condition(cfd->U, cfd->Uprev, cfd->Uin, runtime->dt, mesh->dx, mesh->dy, mesh->dz, sz, GUIDE_CELL, mpi);
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