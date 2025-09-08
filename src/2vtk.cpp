#include <fstream>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <string>
#include <vector>
#include <cassert>
#include <vtkNew.h>
#include <vtkStructuredGrid.h>
#include <vtkRectilinearGrid.h>
#include <vtkXMLRectilinearGridWriter.h>
#include <vtkFloatArray.h>
#include <vtkPointData.h>
#include <vtkUnsignedIntArray.h>
#include "argparse.hpp"
#include "json.hpp"
#include "util.h"
#include "io.h"

using namespace std;
using json = nlohmann::json;

Int vtkindex(Int i, Int j, Int k, Int m, Int size[3], Int dim) {
    return k*size[0]*size[1]*dim + j*size[0]*dim + i*dim + m;
}

void read_binary(
    const string &path,
    vtkNew<vtkRectilinearGrid> &grid,
    bool skip_gc = false
) {
    /** read header data */
    ifstream ifs(path, ios::binary);
    Header header;
    header.read(ifs);
    Int *sz = header.size;
    Int gc = skip_gc? header.gc : 0;
    Int size[] = {sz[0]-2*gc, sz[1]-2*gc, sz[2]-2*gc};
    Int var_count = header.var_count;
    auto &var_dim = header.var_dim;
    auto &var_name = header.var_name;
    header.print_info();

    grid->SetDimensions(size[0], size[1], size[2]);
    const Int cnt = sz[0]*sz[1]*sz[2];
    const Int count = size[0]*size[1]*size[2];

    /** read coordinates */
    vector<Real> x(sz[0]), y(sz[1]), z(sz[2]);

    ifs.read((char*)x.data(), sz[0]*sizeof(Real));
    ifs.read((char*)y.data(), sz[1]*sizeof(Real));
    ifs.read((char*)z.data(), sz[2]*sizeof(Real));

    vtkNew<vtkFloatArray> xv, yv, zv;

    xv->SetNumberOfValues(size[0]);
    for (Int i = 0; i < size[0]; i ++) {
        xv->SetValue(i, x[i+gc]);
    }
    grid->SetXCoordinates(xv);

    yv->SetNumberOfValues(size[1]);
    for (Int j = 0; j < size[1]; j ++) {
        yv->SetValue(j, y[j+gc]);
    }
    grid->SetYCoordinates(yv);

    zv->SetNumberOfValues(size[2]);
    for (Int k = 0; k < size[2]; k ++) {
        zv->SetValue(k, z[k+gc]);
    }
    grid->SetZCoordinates(zv);

    /** read vars */
    for (Int v = 0; v < var_count; v ++) {
        vtkNew<vtkFloatArray> varv;
        varv->SetNumberOfComponents(var_dim[v]);
        varv->SetNumberOfTuples(count);
        varv->SetName(var_name[v].c_str());

        vector<Real> var(cnt*var_dim[v]);
        ifs.read((char*)var.data(), cnt*var_dim[v]*sizeof(Real));
        assert(ifs.gcount() == cnt*var_dim[v]*sizeof(Real));
        for (Int i = 0; i < size[0]; i ++) {
        for (Int j = 0; j < size[1]; j ++) {
        for (Int k = 0; k < size[2]; k ++) {
            for (Int m = 0; m < var_dim[v]; m ++) {
                Int component_id = index(i+gc, j+gc, k+gc, sz)*var_dim[v] + m;
                Int vtk_id = vtkindex(i, j, k, m, size, var_dim[v]);
                varv->SetValue(vtk_id, var[component_id]);
            }
        }}}

        grid->GetPointData()->AddArray(varv);
    }

    ifs.close();
}

int main(int argc, char *argv[]) {
    argparse::ArgumentParser parser;
    parser.add_argument("path").help("file path");
    parser.add_argument("--skip-gc", "-sg").flag().help("skip guide cells");
    parser.parse_args(argc, argv);
    vtkNew<vtkRectilinearGrid> grid;
    vtkNew<vtkXMLRectilinearGridWriter> writer;
    string ifilename = parser.get<string>("path");
    string ofilename = ifilename + ".vtr";
    writer->SetFileName(ofilename.c_str());
    writer->SetCompressionLevel(1);
    writer->SetInputData(grid);

    cout << "read from " << ifilename << endl;
    read_binary(ifilename, grid, (parser["-sg"]==true));

    writer->Write();
    cout << "write to " << ofilename << endl;
}
