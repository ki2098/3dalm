#include "solver.h"

using namespace std;

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    Solver solver;
    string setup_path(argv[1]);
    solver.initialize(setup_path);
    Int *size = solver.size;

    OutHandler ohandler;
    ohandler.set_size(solver.size, solver.gc);

    ohandler.set_var(
        {solver.cfd.U[0], solver.cfd.p, solver.cfd.div, solver.cfd.q},
        {3, 1, 1, 1},
        {"U", "p", "div", "q"}
    );

    for (; solver.rt.step < solver.rt.max_step;) {
        solver.main_loop_once();
    }    

    calc_q(
        solver.cfd.U, solver.cfd.q,
        solver.mesh.x, solver.mesh.y, solver.mesh.z,
        solver.size, solver.gc, &solver.mpi
    );
    ohandler.update_host();

    string filename = make_rank_binary_filename(solver.output_prefix, solver.mpi.rank, solver.rt.step);
    write_binary(
        filename, &ohandler,
        solver.mesh.x, solver.mesh.y, solver.mesh.z
    );
    json slice_json = {{"step", solver.rt.step}, {"time", solver.rt.get_time()}};
    solver.snapshot_json.push_back(slice_json);

    solver.finalize();
    MPI_Finalize();
}