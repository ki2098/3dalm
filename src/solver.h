#pragma once

#include <string>
#include <openacc.h>
#include "json.hpp"
#include "io.h"
#include "bc.h"
#include "cfd.h"
#include "eq.h"

using json = nlohmann::json;

static void calc_partition(
    Int gsize[3], Int size[3], Int offset[3], Int gc,
    MpiInfo *mpi
) {
    size[1] = gsize[1];
    size[2] = gsize[2];
    offset[1] = 0;
    offset[2] = 0;

    Int inner_x_count = gsize[0] - 2*gc;
    Int segment_len = inner_x_count/mpi->size;
    Int leftover = inner_x_count%mpi->size;

    size[0] = segment_len + 2*gc;
    if (mpi->rank < leftover) {
        size[0] ++;
    }
    offset[0] = segment_len*mpi->rank;
    if (mpi->rank < leftover) {
        offset[0] += mpi->rank;
    } else {
        offset[0] += leftover;
    }
}

static void set_turbulence_generating_grid(
    Real solid[],
    Real x[], Real y[], Real z[],
    Real dx[], Real dy[], Real dz[],
    Real thickness, Real spacing, Real placement,
    Int size[3], Int gc
) {
    Int len = size[0]*size[1]*size[2];
#pragma acc kernels loop independent collapse(3) \
present(solid[:len]) \
present(x[:size[0]], y[:size[1]], z[:size[2]]) \
present(dx[:size[0]], dy[:size[1]], dz[:size[2]]) \
copyin(size[:3])
    for (Int i = gc; i < size[0] - gc; i ++) {
    for (Int j = gc; j < size[1] - gc; j ++) {
    for (Int k = gc; k < size[2] - gc; k ++) {
        Real xc = x[i];
        Real yc = y[j];
        Real zc = z[k];
        Real dxc = dx[i];
        Real dyc = dy[j];
        Real dzc = dz[k];
        Real x_intersection = get_intersection(
            xc - 0.5*dxc,
            xc + 0.5*dxc,
            placement - 0.5*thickness,
            placement + 0.5*thickness
        );
        Real y_dist = fabs(yc);
        Real z_dist = fabs(zc);

        Int bar_j_nearest = round(y_dist/spacing);
        Real bar_y = bar_j_nearest*spacing;
        Real y_intersection = get_intersection(
            y_dist - 0.5*dyc,
            y_dist + 0.5*dyc,
            bar_y - 0.5*thickness,
            bar_y + 0.5*thickness
        );

        Int bar_k_nearest = round(z_dist/spacing);
        Real bar_z = bar_k_nearest*spacing;
        Real z_intersection = get_intersection(
            z_dist - 0.5*dzc,
            z_dist + 0.5*dzc,
            bar_z - 0.5*thickness,
            bar_z + 0.5*thickness
        );

        Real occupied = (
            y_intersection*dzc +
            z_intersection*dyc -
            y_intersection*z_intersection
        )*x_intersection;
        
        solid[index(i, j, k, size)] = occupied/(dxc*dyc*dzc);
    }}}
}

struct Runtime {
    Int step = 0, max_step;
    Real dt;

    Real get_time() {
        return dt*step;
    }

    void initialize(Int max_step, Real dt) {
        this->max_step = max_step;
        this->dt = dt;

        // printf("RUNTIME INFO\n");
        // printf("\tmax step = %ld\n", this->max_step);
        // printf("\tdt = %lf\n", this->dt);
    }
};

struct Mesh {
    Real *x, *y, *z, *dx, *dy, *dz;

    void initialize_from_path(std::string path, Int size[3], Int gc, MpiInfo *mpi) {
        build_mesh(path, x, y, z, dx, dy, dz, size, gc, mpi);

#pragma acc enter data \
copyin(x[:size[0]], y[:size[1]], z[:size[2]]) \
copyin(dx[:size[0]], dy[:size[1]], dz[:size[2]])

        // printf("MESH INFO\n");
        // printf("\tfolder = %s\n", path.c_str());
        // printf("\tsize = (%ld %ld %ld)\n", size[0], size[1], size[2]);
        // printf("\tguide cell = %ld\n", gc);
    }

    void initialize_from_global_mesh(Mesh *gmesh, Int gsize[3], Int size[3], Int offset[3], Int gc, MpiInfo *mpi) {
        // size[1] = gsize[1];
        // size[2] = gsize[2];
        // offset[1] = 0;
        // offset[2] = 0;

        // Int inner_x_count = gsize[0] - 2*gc;
        // Int segment_len = inner_x_count/mpi->size;
        // Int leftover = inner_x_count%mpi->size;

        // size[0] = segment_len + 2*gc;
        // if (mpi->rank < leftover) {
        //     size[0] ++;
        // }
        // offset[0] = segment_len*mpi->rank;
        // if (mpi->rank < leftover) {
        //     offset[0] += mpi->rank;
        // } else {
        //     offset[0] += leftover;
        // }
        calc_partition(gsize, size, offset, gc, mpi);

        x = new Real[size[0]];
        dx = new Real[size[0]];
        for (Int i = 0; i < size[0]; i ++) {
            x[i] = gmesh->x[i + offset[0]];
            dx[i] = gmesh->dx[i + offset[0]];
        }

        y = new Real[size[1]];
        dy = new Real[size[1]];
        for (Int j = 0; j < size[1]; j ++) {
            y[j] = gmesh->y[j + offset[1]];
            dy[j] = gmesh->dy[j + offset[1]];
        }

        z = new Real[size[2]];
        dz = new Real[size[2]];
        for (Int k = 0; k < size[2]; k ++) {
            z[k] = gmesh->z[k + offset[2]];
            dz[k] = gmesh->dz[k + offset[2]];
        }

#pragma acc enter data \
copyin(x[:size[0]], y[:size[1]], z[:size[2]]) \
copyin(dx[:size[0]], dy[:size[1]], dz[:size[2]]) 
    }

    void finalize(Int size[3]) {
        delete[] x;
        delete[] y;
        delete[] z;
        delete[] dx;
        delete[] dy;
        delete[] dz;

#pragma acc exit data \
delete(x[:size[0]], y[:size[1]], z[:size[2]]) \
delete(dx[:size[0]], dy[:size[1]], dz[:size[2]])
    }
};

struct Cfd {
    Real (*U)[3], (*Uold)[3];
    Real (*JU)[3];
    Real *p, *nut, *q, *div, *solid;
    Real Uin[3];
    Real Re, Cs;
    Real avg_div, max_cfl;

    void initialize(Real Uin[3], Real Re, Real Cs, Int size[3]) {
        this->Uin[0] = Uin[0];
        this->Uin[1] = Uin[1];
        this->Uin[2] = Uin[2];
        this->Re = Re;
        this->Cs = Cs;

        Int len = size[0]*size[1]*size[2];
        U = new Real[len][3];
        Uold = new Real[len][3];
        JU = new Real[len][3];
        p = new Real[len];
        nut = new Real[len];
        q = new Real[len];
        div = new Real[len];
        solid = new Real[len];

#pragma acc enter data \
create(U[:len], Uold[:len], JU[:len], p[:len], q[:len], nut[:len], div[:len], solid[:len])

        // printf("CFD INFO\n");
        // printf("\tRe = %lf\n", this->Re);
        // printf("\tCs = %lf\n", this->Cs);
        // printf("\tUin = (%lf %lf %lf)\n", this->Uin[0], this->Uin[1], this->Uin[2]);
    }

    void finalize(Int size[3]) {
        delete[] U;
        delete[] Uold;
        delete[] JU;
        delete[] p;
        delete[] nut;
        delete[] q;
        delete[] div;
        delete[] solid;

        Int len = size[0]*size[1]*size[2];
#pragma acc exit data \
delete(U[:len], Uold[:len], JU[:len], p[:len], q[:len], nut[:len], div[:len], solid[:len])
    }
};

struct Eq {
    Real (*A)[7];
    Real *b, *r;
    Real *r0, *p, *pp, *q, *s, *ss, *t, *tmp;
    Int it, max_it;
    Real err, tol;
    Real max_diag;
    Real relax_rate = 1.2;
    Int pc_max_it = 3;

    void initialize(Int max_it, Real tol, Int size[3]) {
        this->max_it = max_it;
        this->tol = tol;

        Int len = size[0]*size[1]*size[2];
        A = new Real[len][7];
        b = new Real[len];
        r = new Real[len];
        r0 = new Real[len];
        p = new Real[len];
        pp = new Real[len];
        q = new Real[len];
        s = new Real[len];
        ss = new Real[len];
        t = new Real[len];
        tmp = new Real[len];

#pragma acc enter data \
create(A[:len], b[:len], r[:len]) \
create(r0[:len], p[:len], pp[:len], q[:len], s[:len], ss[:len], t[:len], tmp[:len])

        // printf("EQ INFO\n");
        // printf("\tmax iteration = %ld\n", this->max_it);
        // printf("\ttolerance = %lf\n", this->tol);
    }

    void finalize(Int size[3]) {
        delete[] A;
        delete[] b;
        delete[] r;
        delete[] r0;
        delete[] p;
        delete[] pp;
        delete[] q;
        delete[] s;
        delete[] ss;
        delete[] t;
        delete[] tmp;

        Int len = size[0]*size[1]*size[2];
#pragma acc exit data \
delete(A[:len], b[:len], r[:len]) \
delete(r0[:len], p[:len], pp[:len], q[:len], s[:len], ss[:len], t[:len], tmp[:len])
    }
};

struct Solver {
    Int gsize[3];
    Int size[3];
    Int offset[3];
    Int gc = 2;

    MpiInfo mpi;
    Runtime rt;
    Mesh gmesh, mesh;
    Cfd cfd;
    Eq eq;

    json setup_json;
    json snapshot_json = {};
    std::string output_prefix;

    void initialize(std::string path) {
        MPI_Comm_size(MPI_COMM_WORLD, &mpi.size);
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi.rank);

        int gpu_count = acc_get_num_devices(acc_device_nvidia);
        int gpu_id = mpi.rank%gpu_count;
        acc_set_device_num(gpu_id, acc_device_nvidia);

        std::ifstream setup_file(path);
        setup_json = json::parse(setup_file);

        auto &rt_json = setup_json["runtime"];
        Real dt = rt_json["dt"];
        Real total_time = rt_json["time"];
        rt.initialize(total_time/dt, dt);

        auto &output_json = setup_json["output"];
        output_prefix = output_json["prefix"];

        auto &tgg_json = setup_json["turbulence_grid"];
        Real tgg_thickness = tgg_json["thickness"];
        Real tgg_spacing = tgg_json["spacing"];
        Real tgg_placement = tgg_json["placement"];

        auto &mesh_json = setup_json["mesh"];
        std::string mesh_path = mesh_json["path"];
        gmesh.initialize_from_path(mesh_path, gsize, gc, &mpi);
        mesh.initialize_from_global_mesh(&gmesh, gsize, size, offset, gc, &mpi);

        // printf("%d mesh OK\n", mpi.rank);

        auto &inflow_json = setup_json["inflow"];
        auto &cfd_json = setup_json["cfd"];
        Real Uin[3];
        Uin[0] = inflow_json["value"][0];
        Uin[1] = inflow_json["value"][1];
        Uin[2] = inflow_json["value"][2];
        Real Re = cfd_json["Re"];
        Real Cs = cfd_json["Cs"];
        cfd.initialize(Uin, Re, Cs, size);

        // printf("%d cfd OK\n", mpi.rank);

        auto &eq_json = setup_json["eq"];
        Real tol = eq_json["tolerance"];
        Real max_it = eq_json["max_iteration"];
        eq.initialize(max_it, tol, size);

        // printf("%d eq OK\n", mpi.rank);

        eq.max_diag = build_A(
            eq.A,
            mesh.x, mesh.y, mesh.z,
            mesh.dx, mesh.dy, mesh.dz,
            gsize, size, offset, gc,
            &mpi
        );

        // printf("%d eq A OK\n", mpi.rank);

        Int len = size[0]*size[1]*size[2];
        fill_array(cfd.U, cfd.Uin, len);
        fill_array(cfd.Uold, cfd.Uin, len);
        fill_array(cfd.p, 0., len);

        set_turbulence_generating_grid(
            cfd.solid,
            mesh.x, mesh.y, mesh.z,
            mesh.dx, mesh.dy, mesh.dz,
            tgg_thickness, tgg_spacing, tgg_placement,
            size, gc
        );

        set_solid_U(
            cfd.U, cfd.solid,
            size, gc
        );

        apply_Ubc(
            cfd.U, cfd.Uold, cfd.Uin,
            mesh.dx, mesh.dy, mesh.dz,
            rt.dt,
            size, gc,
            &mpi
        );

        // printf("%d Ubc OK\n", mpi.rank);

        interpolate_JU(
            cfd.U, cfd.JU,
            mesh.dx, mesh.dy, mesh.dz,
            size, gc,
            &mpi
        );

        // printf("%d JU OK\n", mpi.rank);

        apply_JUbc(
            cfd.JU, cfd.Uin,
            mesh.dy, mesh.dz,
            size, gc,
            &mpi
        );

        // printf("%d JUbc OK\n", mpi.rank);

        calc_nut(
            cfd.U, cfd.nut,
            mesh.x, mesh.y, mesh.z,
            mesh.dx, mesh.dy, mesh.dz,
            cfd.Cs,
            size, gc,
            &mpi
        );

        // printf("%d nut OK\n", mpi.rank);

        calc_divergence(
            cfd.JU, cfd.div,
            mesh.dx, mesh.dy, mesh.dz,
            size, gc,
            &mpi
        );

        // printf("%d div OK\n", mpi.rank);

        Int effective_count = (gsize[0] - 2*gc)*(gsize[1] - 2*gc)*(gsize[2] - 2*gc);
        cfd.avg_div = calc_l2_norm(cfd.div, size, gc, &mpi)/sqrt(effective_count);
        cfd.max_cfl = calc_max_cfl(cfd.U, mesh.dx, mesh.dy, mesh.dz, rt.dt, size, gc, &mpi);

        // printf("%d ||div|| OK\n", mpi.rank);
        if (mpi.rank == 0) {
            printf("SETUP INFO\n");
            printf("\tpath %s\n", path.c_str());

            printf("DEVICE INFO\n");
            printf("\tnumber of GPUs = %d\n", gpu_count);

            printf("MESH INFO\n");
            printf("\tpath = %s\n", mesh_path.c_str());
            printf("\tglobal size = (%ld %ld %ld)\n", gsize[0], gsize[1], gsize[2]);
            printf("\tguide cell = %ld\n", gc);

            printf("RUNTIME INFO\n");
            printf("\tdt = %lf\n", rt.dt);
            printf("\tmax step = %ld\n", rt.max_step);

            printf("OUTPUT INFO\n");
            printf("\tprefix = %s\n", output_prefix.c_str());

            printf("TURBULENCE GENERATING GRID INFO\n");
            printf("\tthickness = %lf\n", tgg_thickness);
            printf("\tspacing = %lf\n", tgg_spacing);
            printf("\tplacement = %lf\n", tgg_placement);

            printf("CFD INFO\n");
            printf("\tRe = %lf\n", cfd.Re);
            printf("\tCs = %lf\n", cfd.Cs);
            printf("\tinitial div(U) = %e\n", cfd.avg_div);
            printf("\tinitial max cfd = %e\n", cfd.max_cfl);

            printf("EQ INFO\n");
            printf("\tmax iteration = %ld\n", eq.max_it);
            printf("\ttolerance = %lf\n", eq.tol);
            printf("\tmax A diag = %lf\n", eq.max_diag);

            write_mesh(
                output_prefix + "_mesh.txt",
                gmesh.x, gmesh.y, gmesh.z,
                gmesh.dx, gmesh.dy, gmesh.dz,
                gsize, gc
            );

            std::ofstream json_output(output_prefix + ".json");
            json_output << std::setw(2) << setup_json;
            json_output.close();

            json part_json;
            part_json["size"] = {gsize[0], gsize[1], gsize[2]};
            part_json["gc"] = gc;
            part_json["partition"] = {};
            for (Int rank = 0; rank < mpi.size; rank ++) {
                json rank_json;
                Int rank_size[3], rank_offset[3];
                MpiInfo rank_info;
                rank_info.size = mpi.size;
                rank_info.rank = rank;
                calc_partition(gsize, rank_size, rank_offset, gc, &rank_info);
                rank_json["size"] = {rank_size[0], rank_size[1], rank_size[2]};
                rank_json["offset"] = {rank_offset[0], rank_offset[1], rank_offset[2]};
                rank_json["rank"] = rank;
                part_json["partition"].push_back(rank_json);
            }
            std::ofstream part_output(output_prefix + "_partition.json");
            part_output << std::setw(2) << part_json;
            part_output.close();
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        for (Int rank = 0; rank < mpi.size; rank ++) {
            if (mpi.rank == rank) {
                printf("PROC INFO %d/%d\n", mpi.rank, mpi.size);
                printf("\tsize = (%ld %ld %ld)\n", size[0], size[1], size[2]);
                printf("\toffset = (%ld %ld %ld)\n", offset[0], offset[1], offset[2]);
                printf("\tGPU id = %d\n", gpu_id);
            }
            fflush(stdout);
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    void finalize() {
        if (mpi.rank == 0) {
            std::ofstream snapshot_output(output_prefix + "_snapshot.json");
            snapshot_output << std::setw(2) << snapshot_json;
            snapshot_output.close();
        }
        gmesh.finalize(gsize);
        mesh.finalize(size);
        cfd.finalize(size);
        eq.finalize(size);
    }

    void main_loop_once() {
        Int len = size[0]*size[1]*size[2];

        cpy_array(cfd.Uold, cfd.U, len);

        // printf("1\n");

        calc_intermediate_U(
            cfd.U, cfd.Uold, cfd.JU, cfd.nut,
            mesh.x, mesh.y, mesh.z,
            mesh.dx, mesh.dy, mesh.dz,
            cfd.Re, rt.dt,
            size, gc,
            &mpi
        );

        interpolate_JU(
            cfd.U, cfd.JU,
            mesh.dx, mesh.dy, mesh.dz,
            size, gc,
            &mpi
        );

        apply_JUbc(
            cfd.JU, cfd.Uin,
            mesh.dy, mesh.dz,
            size, gc,
            &mpi
        );

        // printf("2\n");

        calc_poisson_rhs(
            cfd.JU, eq.b,
            mesh.dx, mesh.dy, mesh.dz,
            rt.dt, eq.max_diag,
            size, gc,
            &mpi
        );

        // printf("3\n");

        // run_sor(
        //     eq.A, cfd.p, eq.b, eq.r,
        //     eq.relax_rate, eq.it, eq.max_it, eq.err, eq.tol,
        //     gsize, size, offset, gc,
        //     &mpi
        // );

        run_pbicgstab(
            eq.A, cfd.p, eq.b, eq.r,
            eq.r0, eq.p, eq.pp, eq.q, eq.s, eq.ss, eq.t, eq.tmp,
            eq.it, eq.max_it, eq.err, eq.tol, eq.pc_max_it,
            gsize, size, gc,
            &mpi
        );

        // run_pbicgstab(
        //     eq.A, cfd.p, eq.b, eq.r,
        //     eq.r0, eq.p, eq.pp, eq.q, eq.s, eq.ss, eq.t,
        //     eq.it, eq.max_it, eq.err, eq.tol, eq.relax_rate, eq.pc_max_it,
        //     gsize, size, offset, gc,
        //     &mpi
        // );

        // printf("4\n");

        apply_pbc(
            cfd.Uold, cfd.p, cfd.nut,
            mesh.z, mesh.dx, mesh.dy, mesh.dz,
            cfd.Re,
            size, gc,
            &mpi
        );

        // printf("5\n");

        project_p(
            cfd.U, cfd.JU, cfd.p,
            mesh.x, mesh.y, mesh.z,
            mesh.dx, mesh.dy, mesh.dz,
            rt.dt,
            size, gc,
            &mpi
        );

        set_solid_U(
            cfd.U, cfd.solid,
            size, gc
        );

        // printf("6\n");

        apply_Ubc(
            cfd.U, cfd.Uold, cfd.Uin,
            mesh.dx, mesh.dy, mesh.dz,
            rt.dt,
            size, gc,
            &mpi
        );

        apply_JUbc(
            cfd.JU, cfd.Uin,
            mesh.dy, mesh.dz,
            size, gc,
            &mpi
        );

        // printf("7\n");

        calc_nut(
            cfd.U, cfd.nut,
            mesh.x, mesh.y, mesh.z,
            mesh.dx, mesh.dy, mesh.dz,
            cfd.Cs,
            size, gc,
            &mpi
        );

        calc_divergence(
            cfd.JU, cfd.div,
            mesh.dx, mesh.dy, mesh.dz,
            size, gc,
            &mpi
        );

        // printf("8\n");

        Int effective_count = (gsize[0] - 2*gc)*(gsize[1] - 2*gc)*(gsize[2] - 2*gc);
        cfd.avg_div = calc_l2_norm(cfd.div, size, gc, &mpi)/sqrt(effective_count);

        cfd.max_cfl = calc_max_cfl(cfd.U, mesh.dx, mesh.dy, mesh.dz, rt.dt, size, gc, &mpi);

        // printf("9\n");

        rt.step ++;

        if (mpi.rank == 0) {
            printf("%ld %e %ld %e %e %e\n", rt.step, rt.get_time(), eq.it, eq.err, cfd.avg_div, cfd.max_cfl);
        }
    }
};