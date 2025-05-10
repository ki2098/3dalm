#include <string>
#include "json.hpp"
#include "io.h"
#include "util.h"

using namespace std;
using json = nlohmann::json;


void load_mesh(
    string path,
    Int size[3], Int &gc,
    vector<Real> &x, vector<Real> &y, vector<Real> &z
) {
    ifstream mesh_file(path);
    Real dummy;
    mesh_file >> size[0] >> size[1] >> size[2] >> gc;
    x.resize(size[0]);
    y.resize(size[1]);
    z.resize(size[2]);
    for (Int i = 0; i < size[0]; i ++) {
        mesh_file >> x[i] >> dummy;
    }
    for (Int j = 0; j < size[1]; j ++) {
        mesh_file >> y[j] >> dummy;
    }
    for (Int k = 0; k < size[2]; k ++) {
        mesh_file >> z[k] >> dummy;
    }

    mesh_file.close();
}

int main(int argc, char *argv[]) {
    string prefix(argv[1]);
    string part_path = prefix + "_partition.json";
    string snapshot_path = prefix + "_snapshot.json";
    string mesh_path = prefix + "_mesh.txt";
    ifstream part_file(part_path);
    ifstream snapshot_file(snapshot_path);
    auto part_json = json::parse(part_file);
    auto snapshot_json = json::parse(snapshot_file);
    part_file.close();
    snapshot_file.close();

    Header gheader;

    if (snapshot_json.size() == 0) {
        cout << "no data to be reconstructed" << endl;
        return 0;
    }

    Int peek_step = snapshot_json[0]["step"];
    string peek_filename = make_rank_binary_filename(prefix, 0, peek_step);
    ifstream peek_ifs(peek_filename);
    gheader.read(peek_ifs);
    peek_ifs.close();

    vector<Real> gx, gy, gz;
    load_mesh(mesh_path, gheader.size, gheader.gc, gx, gy, gz);

    printf("GLOBAL HEADER INFO\n");
    gheader.print_info();
}