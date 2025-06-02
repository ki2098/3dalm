#pragma once

#include <filesystem>
#include <string>
#include <openacc.h>
#include <array>
#include "json.hpp"
#include "argparse.hpp"
#include "io.h"
#include "bc.h"
#include "cfd.h"
#include "eq.h"

using json = nlohmann::json;

// static void delete_prev_directory(std::string path) {
//     if (!std::filesystem::is_directory(path)) {
//         return;
//     }

//     if (std::filesystem::remove_all(path)) {
//         printf("delete directory %s\n", path.c_str());
//     }
// }

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
    Real tgg_thick, Real tgg_mesh, Real tgg_bar, Real tgg_x,
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
        // Real x_intersection = get_intersection(
        //     xc - 0.5*dxc,
        //     xc + 0.5*dxc,
        //     tgg_x - 0.5*tgg_thick,
        //     tgg_x + 0.5*tgg_thick
        // );
        Real y_dist = fabs(yc);
        Real z_dist = fabs(zc);

        // Int bar_j_nearest = round(y_dist/tgg_mesh);
        // Real bar_y = bar_j_nearest*tgg_mesh;
        // Real y_intersection = get_intersection(
        //     y_dist - 0.5*dyc,
        //     y_dist + 0.5*dyc,
        //     bar_y - 0.5*tgg_bar,
        //     bar_y + 0.5*tgg_bar
        // );

        // Int bar_k_nearest = round(z_dist/tgg_mesh);
        // Real bar_z = bar_k_nearest*tgg_mesh;
        // Real z_intersection = get_intersection(
        //     z_dist - 0.5*dzc,
        //     z_dist + 0.5*dzc,
        //     bar_z - 0.5*tgg_bar,
        //     bar_z + 0.5*tgg_bar
        // );

        // Real occupied = (
        //     y_intersection*dzc +
        //     z_intersection*dyc -
        //     y_intersection*z_intersection
        // )*x_intersection;

        Real x_intersec_vertical = get_intersection(
            xc - 0.5*dxc, xc + 0.5*dxc,
            tgg_x - tgg_thick, tgg_x
        );
        Int bar_j_nearest = round(y_dist/tgg_mesh);
        Real bar_y = bar_j_nearest*tgg_mesh;
        Real y_intersec = get_intersection(
            y_dist - 0.5*dyc, y_dist + 0.5*dyc,
            bar_y - 0.5*tgg_bar, bar_y + 0.5*tgg_bar
        );

        Real x_intersec_horizontal = get_intersection(
            xc - 0.5*dxc, xc + 0.5*dxc,
            tgg_x - 2*tgg_thick, tgg_x - tgg_thick
        );
        
        Int bar_k_nearest = round(z_dist/tgg_mesh);
        Real bar_z = bar_k_nearest*tgg_mesh;
        Real z_intersec = get_intersection(
            z_dist - 0.5*dzc, z_dist + 0.5*dzc,
            bar_z - 0.5*tgg_bar, bar_z + 0.5*tgg_bar
        );

        Real occupied = x_intersec_vertical*y_intersec*dzc + x_intersec_horizontal*z_intersec*dyc;
        solid[index(i, j, k, size)] = occupied/(dxc*dyc*dzc);
    }}}
}

struct Runtime {
    Int step = 0, max_step;
    Real dt;
    Int output_start_step;
    Int output_interval_step;
    Int tavg_start_step;

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

    void initialize_from_file(const std::string &path, Int size[3], Int &gc, MpiInfo *mpi) {
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
    Real (*U)[3], (*Uold)[3], (*Utavg)[3];
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
        U = new Real[len][3]();
        Uold = new Real[len][3]();
        Utavg = new Real[len][3]();
        JU = new Real[len][3]();
        p = new Real[len]();
        nut = new Real[len]();
        q = new Real[len]();
        div = new Real[len]();
        solid = new Real[len]();

#pragma acc enter data \
create(U[:len], Uold[:len], Utavg[:len], JU[:len], p[:len], q[:len], nut[:len], div[:len], solid[:len])

        // printf("CFD INFO\n");
        // printf("\tRe = %lf\n", this->Re);
        // printf("\tCs = %lf\n", this->Cs);
        // printf("\tUin = (%lf %lf %lf)\n", this->Uin[0], this->Uin[1], this->Uin[2]);
    }

    void finalize(Int size[3]) {
        delete[] U;
        delete[] Uold;
        delete[] Utavg;
        delete[] JU;
        delete[] p;
        delete[] nut;
        delete[] q;
        delete[] div;
        delete[] solid;

        Int len = size[0]*size[1]*size[2];
#pragma acc exit data \
delete(U[:len], Uold[:len], Utavg[:len], JU[:len], p[:len], q[:len], nut[:len], div[:len], solid[:len])
    }
};

struct Eq {
    std::string method, pc_method;

    Real (*A)[7];
    Real *b, *r;
    Real *r0, *p, *pp, *q, *s, *ss, *t, *tmp;
    Int it, max_it;
    Real err, tol;
    Real max_diag;
    Real relax_rate;
    Real pc_relax_rate;
    Int pc_max_it;

    void initialize(Int max_it, Real tol, Int size[3], const std::string &method) {
        this->max_it = max_it;
        this->tol = tol;
        this->method = method;

        Int len = size[0]*size[1]*size[2];
        A = new Real[len][7]();
        b = new Real[len]();
        r = new Real[len]();
        r0 = new Real[len]();
        p = new Real[len]();
        pp = new Real[len]();
        q = new Real[len]();
        s = new Real[len]();
        ss = new Real[len]();
        t = new Real[len]();
        tmp = new Real[len]();

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

struct Probe {
    Int i, j, k;
    Real x, y, z;
    std::string path;
    bool active = false;

    Int max_rec_cnt;
    Int cur_rec_cnt = 0;
    std::vector<std::array<float, 4>> recs;

    void initialize(
        const std::string &path,
        Int i, Int j, Int k, Real x, Real y, Real z, Int max_rec_cnt = 10000
    ) {
        this->path = path;
        this->i = i;
        this->j = j;
        this->k = k;
        this->x = x;
        this->y = y;
        this->z = z;
        std::ofstream ofs(this->path);
        ofs << "# " << this->x << " ";
        ofs << this->y << " ";
        ofs << this->z << std::endl;
        ofs << "t,u,v,w" << std::endl;
        active = true;

        this->max_rec_cnt = max_rec_cnt;
        recs = std::vector<std::array<float, 4>>(max_rec_cnt);
    }

    void add_record(float t, const std::array<float, 3> &U) {
        recs[cur_rec_cnt] = {t, U[0], U[1], U[2]};
        cur_rec_cnt ++;
        if (cur_rec_cnt == max_rec_cnt) {
            std::ofstream ofs(this->path, std::ios::app);
            for (const auto &rec : recs) {
                ofs << rec[0] << ",";
                ofs << rec[1] << ",";
                ofs << rec[2] << ",";
                ofs << rec[3] << std::endl;
            }
            cur_rec_cnt = 0;
        }
    }

    ~Probe() {
        if (cur_rec_cnt > 0) {
            std::ofstream ofs(this->path, std::ios::app);
            for (int i = 0; i < cur_rec_cnt; i ++) {
                auto &rec = recs[i];
                ofs << rec[0] << ",";
                ofs << rec[1] << ",";
                ofs << rec[2] << ",";
                ofs << rec[3] << std::endl;
            }
        }
    }
};

struct Solver {
    Int gsize[3];
    Int size[3];
    Int offset[3];
    Int gc;

    MpiInfo mpi;
    Runtime rt;
    Mesh gmesh, mesh;
    Cfd cfd;
    Eq eq;

    json setup_json;
    json snapshot_json = {};
    std::filesystem::path case_dir, output_dir;

    OutHandler out_handler, tavg_out_handler;

    std::vector<Probe> probes;

    void initialize(int argc, char **argv) {        
        MPI_Init(&argc, &argv);
        MPI_Comm_size(MPI_COMM_WORLD, &mpi.size);
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi.rank);

        argparse::ArgumentParser parser;
        parser.add_argument("case")
            .help("case directory");
        parser.add_argument("-c", "--clear")
            .flag()
            .help("delete existing output folder");
        parser.parse_args(argc, argv);

        case_dir = std::filesystem::canonical(parser.get<std::string>("case"));
        // if (setup_path.has_parent_path()) {
        //     std::filesystem::current_path(setup_path.parent_path());
        // }

        int gpu_count = acc_get_num_devices(acc_device_nvidia);
        int gpu_id = mpi.rank%gpu_count;
        acc_set_device_num(gpu_id, acc_device_nvidia);

        setup_json = json::parse(std::ifstream(case_dir/"setup.json"));
        MPI_Barrier(MPI_COMM_WORLD);

        auto &rt_json = setup_json["runtime"];
        Real dt = rt_json["dt"];
        Real total_time = rt_json["time"];
        rt.initialize(total_time/dt, dt);

        auto &output_json = setup_json["output"];
        output_dir = case_dir/"output";

        if (mpi.rank == 0) {
            if (parser["-c"] == true) {
                if (std::filesystem::exists(output_dir)) {
                    if (std::filesystem::remove_all(output_dir)) {
                        printf("delete %s\n", output_dir.c_str());
                    }
                }
            }

            if (!std::filesystem::exists(output_dir)) {
                if (std::filesystem::create_directories(output_dir)) {
                    printf("create %s\n", output_dir.c_str());
                }
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);

        Real output_start_time = output_json["output start time"];
        Real output_interval_time = output_json["output interval time"];
        Real tavg_start_time = output_json["time avg start time"];
        rt.output_start_step = output_start_time/rt.dt;
        rt.output_interval_step = output_interval_time/rt.dt;
        rt.tavg_start_step = tavg_start_time/rt.dt;

        Real tbb_bar, tgg_thick, tgg_mesh, tgg_x;
        auto it_tgg_json = setup_json.find("turbulence grid");
        bool there_is_tgg = (it_tgg_json != setup_json.end());
        if (there_is_tgg) {
            auto &tgg_json = *it_tgg_json;
            tgg_thick = tgg_json["thick"];
            tgg_mesh = tgg_json["mesh"];
            tbb_bar = tgg_json["bar"];
            tgg_x = tgg_json["x"];
        }

        auto &mesh_json = setup_json["mesh"];
        auto mesh_path = case_dir/"mesh.txt";
        gmesh.initialize_from_file(mesh_path, gsize, gc, &mpi);
        assert(gc >= 2);
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
        Real max_it = eq_json["max iteration"];
        eq.initialize(max_it, tol, size, eq_json["method"]);
        if (eq.method == "BiCG") {
            auto &pc_json = eq_json["preconditioner"];
            eq.pc_method = pc_json["method"];
            eq.pc_max_it = pc_json["max iteration"];
            if (eq.pc_method == "SOR") {
                eq.pc_relax_rate = pc_json["relaxation rate"];
            }
        } else if (eq.method == "SOR") {
            eq.relax_rate = eq_json["relaxation rate"];
        }

        // printf("%d eq OK\n", mpi.rank);

        auto it_probes_json = setup_json.find("probe");
        if (it_probes_json != setup_json.end()) {
            auto &probes_json = *it_probes_json;
            probes = std::vector<Probe>(probes_json.size());
            for (Int m = 0; m < probes.size(); m ++) {
                auto &probe_json = probes_json[m];
                Real probex = probe_json[0];
                Real probey = probe_json[1];
                Real probez = probe_json[2];
                Real *x = mesh.x;
                Real *y = mesh.y;
                Real *z = mesh.z;

                // for (Int i = gc - 1; i < size[0] - gc; i ++) {
                //     if (x[i] <= probex && x[i + 1] > probex) {
                //         neari = (
                //             probex - x[i] < x[i + 1] - probex
                //         )? i : i + 1;
                //         if (nearest_i >= gc && nearest_i < size[0] - gc) {
                //             probes[m].initialize(
                //                 output_dir/("probe" + std::to_string(m) + ".csv"),
                //                 nearest_i,
                //                 x[nearest_i],
                //                 rt.output_interval_step
                //             );
                //         }
                //         break;
                //     }
                // }
                Int neari = find_nearest_index(x + gc - 1, probex, size[0] - 2*gc + 2) + gc - 1;
                Int nearj = find_nearest_index(y + gc - 1, probey, size[1] - 2*gc + 2) + gc - 1;
                Int neark = find_nearest_index(z + gc - 1, probez, size[2] - 2*gc + 2) + gc - 1;
                if (
                    neari >= gc && neari < size[0] - gc
                &&  nearj >= gc && nearj < size[1] - gc
                &&  neark >= gc && neark < size[2] - gc
                ) {
                    probes[m].initialize(
                        output_dir/("probe" + std::to_string(m) + ".csv"),
                        neari, nearj, neark,
                        x[neari], y[nearj], z[neark],
                        rt.output_interval_step
                    );
                }
            }
        }
        

        /* if (eq.method == "BiCG") {
            eq.max_diag = build_A_for_BiCG(
                eq.A,
                mesh.x, mesh.y, mesh.z,
                mesh.dx, mesh.dy, mesh.dz,
                gsize, size, offset, gc,
                &mpi
            );
        } else if (eq.method == "SOR") {
            eq.max_diag = build_A_for_SOR(
                eq.A,
                mesh.x, mesh.y, mesh.z,
                mesh.dx, mesh.dy, mesh.dz,
                gsize, size, offset, gc,
                &mpi
            );
        } */
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

        if (there_is_tgg)
        {
            set_turbulence_generating_grid(
                cfd.solid,
                mesh.x, mesh.y, mesh.z,
                mesh.dx, mesh.dy, mesh.dz,
                tgg_thick, tgg_mesh, tbb_bar, tgg_x,
                size, gc
            );
            OutHandler tgg_out_handler;
            tgg_out_handler.set_size(size, gc);
            tgg_out_handler.set_var(
                {cfd.solid},
                {1},
                {"solid"}
            );
            tgg_out_handler.update_host();
            write_binary(
                output_dir/make_rank_binary_filename("solid", mpi.rank, rt.step),
                &tgg_out_handler,
                mesh.x, mesh.y, mesh.z
            );
        } else {
            fill_array(cfd.solid, 0., len);
        }
        

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

        out_handler.set_size(size, gc);
        out_handler.set_var(
            {cfd.U[0], cfd.p, cfd.div, cfd.q},
            {3, 1, 1, 1},
            {"U", "p", "div", "q"}
        );

        tavg_out_handler.set_size(size, gc);
        tavg_out_handler.set_var(
            {cfd.Utavg[0]},
            {3},
            {"U"}
        );

        if (mpi.rank == 0) {
            // write_mesh(
            //     output_dir/"mesh.txt",
            //     gmesh.x, gmesh.y, gmesh.z,
            //     gmesh.dx, gmesh.dy, gmesh.dz,
            //     gsize, gc
            // );

            printf("SETUP INFO\n");
            printf("\tcase directory = %s\n", case_dir.c_str());

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
            printf("\tdirectory = %s\n", output_dir.c_str());
            printf("\toutput start step = %ld\n", rt.output_start_step);
            printf("\toutput interval step = %ld\n", rt.output_interval_step);
            printf("\ttime avg start step = %ld\n", rt.tavg_start_step);
            printf("\toutput vars =");
            for (auto &v : out_handler.var_name) {
                printf(" %s", v.c_str());
            }
            printf("\n");
            printf("\ttime avg vars =");
            for (auto &v : tavg_out_handler.var_name) {
                printf(" %s", v.c_str());
            }
            printf("\n");

            printf("TURBULENCE GENERATING GRID INFO\n");
            if (there_is_tgg) {
                printf("\tthickness = %lf\n", tgg_thick);
                printf("\tspacing = %lf\n", tgg_mesh);
                printf("\tplacement = %lf\n", tgg_x);
            } else {
                printf("\tno turbulence generating grid specified\n");
            }

            printf("CFD INFO\n");
            printf("\tRe = %lf\n", cfd.Re);
            printf("\tCs = %lf\n", cfd.Cs);
            printf("\tinitial div(U) = %e\n", cfd.avg_div);
            printf("\tinitial max cfd = %e\n", cfd.max_cfl);

            printf("EQ INFO\n");
            printf("\tmax iteration = %ld\n", eq.max_it);
            printf("\ttolerance = %lf\n", eq.tol);
            printf("\tmax A diag = %lf\n", eq.max_diag);
            printf("\tmethod = %s\n", eq.method.c_str());
            if (eq.method == "SOR") {
                printf("\trelaxation rate = %lf\n", eq.relax_rate);
            } else if (eq.method == "BiCG") {
                printf("PRECONDITIONER INFO\n");
                printf("\tmethod = %s\n", eq.pc_method.c_str());
                printf("\tmax iteration = %ld\n", eq.pc_max_it);
                if (eq.pc_method == "SOR") {
                    printf("\trelaxation rate = %lf\n", eq.pc_relax_rate);
                }
            }

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
            std::ofstream part_output(output_dir/"partition.json");
            part_output << std::setw(2) << part_json;
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        for (Int rank = 0; rank < mpi.size; rank ++) {
            if (mpi.rank == rank) {
                printf("PROC INFO %d/%d\n", mpi.rank, mpi.size);
                printf("\tgsize = (%ld %ld %ld)\n", gsize[0], gsize[1], gsize[2]);
                printf("\tsize = (%ld %ld %ld)\n", size[0], size[1], size[2]);
                printf("\toffset = (%ld %ld %ld)\n", offset[0], offset[1], offset[2]);
                printf("\tGPU id = %d\n", gpu_id);
                printf("\tprobes = \n");
                for (auto &m : probes) {
                    if (m.active) {
                        printf("\t\tposition = (%lf %lf %lf), local index = (%ld %ld %ld)\n",
                        m.x, m.y, m.z,
                        m.i, m.j, m.k);
                    }
                }
                fflush(stdout);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    void finalize() {
        if (mpi.rank == 0) {
            std::ofstream snapshot_output(output_dir/"snapshot.json");
            snapshot_output << std::setw(2) << snapshot_json;
        }
        gmesh.finalize(gsize);
        mesh.finalize(size);
        cfd.finalize(size);
        eq.finalize(size);
        MPI_Finalize();
    }

    void main_loop_once() {
        Int len = size[0]*size[1]*size[2];
        Int effective_count = (gsize[0] - 2*gc)*(gsize[1] - 2*gc)*(gsize[2] - 2*gc);
        
        if (rt.step == 0) {
            goto SKIP_TIME_INTEGRAL;
        }

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

        if (eq.method == "BiCG") {
            if (eq.pc_method == "SOR") {
                run_pbicgstab(
                    eq.A, cfd.p, eq.b, eq.r,
                    eq.r0, eq.p, eq.pp, eq.q, eq.s, eq.ss, eq.t,
                    eq.it, eq.max_it, eq.err, eq.tol, eq.pc_relax_rate, eq.pc_max_it,
                    gsize, size, offset, gc,
                    &mpi
                );
            } else if (eq.pc_method == "Jacobi") {
                run_pbicgstab(
                    eq.A, cfd.p, eq.b, eq.r,
                    eq.r0, eq.p, eq.pp, eq.q, eq.s, eq.ss, eq.t, eq.tmp,
                    eq.it, eq.max_it, eq.err, eq.tol, eq.pc_max_it,
                    gsize, size, gc,
                    &mpi
                );
            }
        } else if (eq.method == "SOR") {
            run_sor(
                eq.A, cfd.p, eq.b, eq.r,
                eq.relax_rate, eq.it, eq.max_it, eq.err, eq.tol,
                gsize, size, offset, gc,
                &mpi
            );
        }

        

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

        calc_q(
            cfd.U, cfd.q,
            mesh.x, mesh.y, mesh.z,
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

        cfd.avg_div = calc_l2_norm(cfd.div, size, gc, &mpi)/sqrt(effective_count);

        cfd.max_cfl = calc_max_cfl(cfd.U, mesh.dx, mesh.dy, mesh.dz, rt.dt, size, gc, &mpi);

        if (mpi.rank == 0) {
            printf("%ld %e %ld %e %e %e\n", rt.step, rt.get_time(), eq.it, eq.err, cfd.avg_div, cfd.max_cfl);
        }

SKIP_TIME_INTEGRAL:
        // printf("9\n");
        if (rt.step >= rt.tavg_start_step) {
            Real weight = 1./(rt.step - rt.tavg_start_step + 1);
            calc_ax_plus_by(
                1. - weight, cfd.Utavg,
                weight, cfd.U,
                cfd.Utavg,
                len
            );
        }

        if (rt.step >= rt.output_start_step) {
            for (auto &m : probes) {
                if (m.active) {
                    Int id = index(m.i, m.j, m.k, size);
#pragma acc update host(cfd.U[id:1])
                    m.add_record(
                        rt.get_time(),
                        {(float)cfd.U[id][0], (float)cfd.U[id][1], (float)cfd.U[id][2]}
                    );
                }
            }

            if ((rt.step - rt.output_start_step)%rt.output_interval_step == 0) {
                out_handler.update_host();
                write_binary(
                    output_dir/make_rank_binary_filename("inst", mpi.rank, rt.step),
                    &out_handler,
                    mesh.x, mesh.y, mesh.z
                );
                json slice_json = {
                    {"step", rt.step},
                    {"time", rt.get_time()}
                };

                if (rt.step >= rt.tavg_start_step) {
                    tavg_out_handler.update_host();
                    write_binary(
                        output_dir/make_rank_binary_filename("tavg", mpi.rank, rt.step),
                        &tavg_out_handler,
                        mesh.x, mesh.y, mesh.z
                    );
                    slice_json["tavg"] = "yes";
                }
                
                snapshot_json.push_back(slice_json);
                if (mpi.rank == 0) {
                    std::ofstream snapshot_output(output_dir/"snapshot.json");
                    snapshot_output << std::setw(2) << snapshot_json;
                }
            }
        }

        rt.step ++;
    }

    bool can_continue() {
        return rt.step <= rt.max_step;
    }
};