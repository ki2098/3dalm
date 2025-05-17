#include "solver.h"

using namespace std;

int main(int argc, char *argv[]) {
    Solver solver;
    solver.initialize(argc, argv);

    // OutHandler ohandler;
    // ohandler.set_size(solver.size, solver.gc);

    // ohandler.set_var(
    //     {solver.cfd.U[0], solver.cfd.p, solver.cfd.div, solver.cfd.q},
    //     {3, 1, 1, 1},
    //     {"U", "p", "div", "q"}
    // );

    while (solver.can_continue()) {
        solver.main_loop_once();
    }    

    // calc_q(
    //     solver.cfd.U, solver.cfd.q,
    //     solver.mesh.x, solver.mesh.y, solver.mesh.z,
    //     solver.size, solver.gc, &solver.mpi
    // );
    // ohandler.update_host();

    // string filename = make_rank_binary_filename(solver.output_prefix, solver.mpi.rank, solver.rt.step);
    // write_binary(
    //     filename, &ohandler,
    //     solver.mesh.x, solver.mesh.y, solver.mesh.z
    // );
    // json slice_json = {{"step", solver.rt.step}, {"time", solver.rt.get_time()}};
    // solver.snapshot_json.push_back(slice_json);

    solver.finalize();
}