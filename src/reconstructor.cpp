#include <string>
#include <vector>
#include <array>
#include <numeric>
#include <cassert>
#include <filesystem>
#include "json.hpp"
#include "io.h"
#include "util.h"

using namespace std;
using json = nlohmann::json;
namespace fs = std::filesystem;

void load_mesh(
    const string &path,
    Int size[3], Int &gc,
    Real *&x, Real *&y, Real *&z
) {
    ifstream mesh_file(path);
    Real dummy;
    mesh_file >> size[0] >> size[1] >> size[2] >> gc;
    x = new Real[size[0]];
    y = new Real[size[1]];
    z = new Real[size[2]];
    for (Int i = 0; i < size[0]; i ++) {
        mesh_file >> x[i] >> dummy;
    }
    for (Int j = 0; j < size[1]; j ++) {
        mesh_file >> y[j] >> dummy;
    }
    for (Int k = 0; k < size[2]; k ++) {
        mesh_file >> z[k] >> dummy;
    }

}

void merge_rank_var(
    Real *gvar, Int gsize[3],
    Real *var, Int size[3], Int offset[3],
    Int gc, Int dim
) {
    Int start[3] = {
        (offset[0] == 0)? 0 : gc,
        (offset[1] == 0)? 0 : gc,
        (offset[2] == 0)? 0 : gc
    };
    for (Int i = start[0]; i < size[0]; i ++) {
    for (Int j = start[1]; j < size[1]; j ++) {
    for (Int k = start[2]; k < size[2]; k ++) {
        Int id = index(i, j, k, size);
        Int gid = index(i + offset[0], j + offset[1], k + offset[2], gsize);
        for (Int m = 0; m < dim; m ++) {
            gvar[gid*dim + m] = var[id*dim + m];
        }
    }}}
}

void reconstruct(const fs::path &path, bool tavg = false) {
    auto snapshot_json = json::parse(std::ifstream(path/"snapshot.json"));
    auto part_json = json::parse(std::ifstream(path/"partition.json"));

    OutHandler out_handler;

    Int peek_step = -1;
    string prefix;
    if (tavg) {
        prefix = "tavg";
        for (auto &peek : snapshot_json) {
            if (peek.value("tavg", "no") == "yes") {
                peek_step = peek["step"];
                break;
            }
        }
    } else {
        prefix = "inst";
        for (auto &peek : snapshot_json) {
            peek_step = peek["step"];
            break;
        }
    }
    if (peek_step == -1) {
        cout << "no data to be reconstructed" << endl;
        return;
    }

    auto peek_path = path/make_rank_binary_filename(prefix, 0, peek_step);
    ifstream peek_ifs(peek_path);
    // gheader.read(peek_ifs);
    out_handler.read(peek_ifs);
    Int var_count = out_handler.var_count;
    auto &var_dim = out_handler.var_dim;
    auto &var_name = out_handler.var_name;

    Real *gx, *gy, *gz;
    load_mesh(path/"mesh.txt", out_handler.size, out_handler.gc, gx, gy, gz);
    auto gsize = out_handler.size;
    Int gc = out_handler.gc;
    Int glen = gsize[0]*gsize[1]*gsize[2];

    printf("GLOBAL HEADER INFO\n");
    out_handler.print_info();

    auto rank_json_list = part_json["partition"];
    int mpi_size = rank_json_list.size();
    vector<Int[3]> size_list(mpi_size), offset_list(mpi_size);
    for (auto &rank_json : rank_json_list) {
        int rank = rank_json["rank"];
        size_list[rank][0] = rank_json["size"][0];
        size_list[rank][1] = rank_json["size"][1];
        size_list[rank][2] = rank_json["size"][2];
        offset_list[rank][0] = rank_json["offset"][0];
        offset_list[rank][1] = rank_json["offset"][1];
        offset_list[rank][2] = rank_json["offset"][2];
        printf("PARTITION %d INFO\n", rank);
        auto size = size_list[rank];
        auto offset = offset_list[rank];
        printf("\tsize = (%ld %ld %ld)\n", size[0], size[1], size[2]);
        printf("\toffset = (%ld %ld %ld)\n", offset[0], offset[1], offset[2]);
    }

    for (auto &snapshot : snapshot_json) {
        if (tavg && snapshot.value("tavg", "no") != "yes") {
            continue;
        }

        Int step = snapshot["step"];
  
        auto result_path = path/make_binary_filename(prefix, step);
        // Real **gdata = new Real*[var_count];
        // for (Int v = 0; v < var_count; v ++) {
        //     gdata[v] = new Real[glen*var_dim[v]];
        // }
        out_handler.var = vector<Real*>(var_count);
        for (Int v = 0; v < var_count; v ++) {
            out_handler.var[v] = new Real[glen*var_dim[v]];
        }

        for (int rank = 0; rank < mpi_size; rank ++) {
            auto rank_file_path = path/make_rank_binary_filename(prefix, rank, step);
            ifstream ifs(rank_file_path);

            Header header;
            header.read(ifs);
            auto size = size_list[rank];
            auto offset = offset_list[rank];
            Int len = size[0]*size[1]*size[2];

            // vector<Real> x(size[0]), y(size[1]), z(size[2]);
            // ifs.read((char*)x.data(), size[0]*sizeof(Real));
            // ifs.read((char*)y.data(), size[1]*sizeof(Real));
            // ifs.read((char*)z.data(), size[2]*sizeof(Real));
            ifs.seekg(
                sizeof(Real)*(size[0] + size[1] + size[2]),
                ifs.cur
            );

            printf("merging %s to %s...", rank_file_path.c_str(), result_path.c_str());

            for (Int v = 0; v < var_count; v ++) {
                Real *var = new Real[len*var_dim[v]];
                ifs.read((char*)var, len*var_dim[v]*sizeof(Real));
                assert(ifs.gcount() == len*var_dim[v]*sizeof(Real));
                merge_rank_var(
                    out_handler.var[v], gsize,
                    var, size, offset,
                    gc, var_dim[v]
                );
                delete[] var;
                printf("%s ", var_name[v].c_str());
            }
            printf("\n");
        }

        write_binary(
            result_path,
            &out_handler, gx, gy, gz
        );

        for (auto &v : out_handler.var) {
            delete[] v;
        }
    }

    delete[] gx;
    delete[] gy;
    delete[] gz;
}

int main(int argc, char *argv[]) {
    printf("reconstruct instantaneous field files\n");
    reconstruct(argv[1]);
    printf("reconstruct time averaged field files\n");
    reconstruct(argv[1], true);
}