#include "solver.h"

using namespace std;

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    Solver solver;
    string setup_path(argv[1]);
    solver.initialize(setup_path);
    Int *size = solver.size;
    Int len = size[0]*size[1]*size[2];

    Header header;
    header.size[0] = solver.size[0];
    header.size[1] = solver.size[1];
    header.size[2] = solver.size[2];
    header.gc = solver.gc;
    header.var_count = 3;
    header.var_dim = {3, 1, 1};
    header.var_name = {"U", "p", "div"};
    Real *var[] = {solver.cfd.U[0], solver.cfd.p, solver.cfd.div};

// #pragma acc update \
// host(solver.cfd.U[:len], solver.cfd.p[:len], solver.cfd.div[:len])
//     // write_csv(
//     //     "data/0.csv",
//     //     var, var_count, var_dim, var_name,
//     //     solver.mesh.x, solver.mesh.y, solver.mesh.z,
//     //     solver.size, solver.gc
//     // );

    for (; solver.rt.step < solver.rt.max_step;) {
        solver.main_loop_once();
    }    

#pragma acc update \
host(solver.cfd.U[:len], solver.cfd.p[:len], solver.cfd.div[:len])
    string filename = make_rank_binary_filename(solver.output_prefix, solver.mpi.rank, solver.rt.step);
    write_binary(
        filename, &header,
        var, solver.mesh.x, solver.mesh.y, solver.mesh.z
    );
    json slice_json = {{"step", solver.rt.step}, {"time", solver.rt.get_time()}};
    solver.snapshot_json.push_back(slice_json);

    solver.finalize();
    MPI_Finalize();
}