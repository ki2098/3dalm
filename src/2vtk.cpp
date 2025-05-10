#include <fstream>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <string>
#include <vector>
#include <vtkNew.h>
#include <vtkStructuredGrid.h>
#include <vtkRectilinearGrid.h>
#include <vtkXMLRectilinearGridWriter.h>
#include <vtkFloatArray.h>
#include <vtkPointData.h>
#include <vtkUnsignedIntArray.h>
#include "json.hpp"
#include "util.h"
#include "io.h"

using namespace std;
using json = nlohmann::json;

Int vtkindex(Int i, Int j, Int k, Int m, Int size[3], Int dim) {
    return k*size[0]*size[1]*dim + j*size[0]*dim + i*dim + m;
}

void read_binary(
    string path,
    vtkNew<vtkRectilinearGrid> &grid
) {
    /** read header data */
    ifstream ifs(path, ios::binary);
    Int size[3], gc, var_count;
    ifs.read((char*)size, 3*sizeof(Int));
    ifs.read((char*)&gc, sizeof(Int));
    ifs.read((char*)&var_count, sizeof(Int));
    vector<Int> var_dim(var_count);
    ifs.read((char*)var_dim.data(), var_count*sizeof(Int));
    vector<string> var_name;
    for (Int v = 0; v < var_count; v ++) {
        Int len;
        ifs.read((char*)&len, sizeof(Int));
        string s(len, '\0');
        ifs.read((char*)s.c_str(), len*sizeof(char));
        var_name.push_back(s);
    }
    printf("size = (%ld %ld %ld)\n", size[0], size[1], size[2]);
    printf("gc = %ld\n", gc);
    printf("var count = %ld\n", var_count);
    printf("var dim = (\n");
    for (int v = 0; v < var_count; v ++) {
        printf("\t%ld\n", var_dim[v]);
    }
    printf(")\n");
    printf("var name = (\n");
    for (auto &s : var_name) {
        printf("\t%s\n", s.c_str());
    }
    printf(")\n");

    grid->SetDimensions(size[0], size[1], size[2]);
    const Int count = size[0]*size[1]*size[2];

    /** read coordinates */
    vector<Real> x(size[0]), y(size[1]), z(size[2]);

    ifs.read((char*)x.data(), size[0]*sizeof(Real));
    ifs.read((char*)y.data(), size[1]*sizeof(Real));
    ifs.read((char*)z.data(), size[2]*sizeof(Real));

    vtkNew<vtkFloatArray> xv, yv, zv;

    xv->SetNumberOfValues(size[0]);
    for (Int i = 0; i < size[0]; i ++) {
        xv->SetValue(i, x[i]);
    }
    grid->SetXCoordinates(xv);

    yv->SetNumberOfValues(size[1]);
    for (Int j = 0; j < size[1]; j ++) {
        yv->SetValue(j, y[j]);
    }
    grid->SetYCoordinates(yv);

    zv->SetNumberOfValues(size[2]);
    for (Int k = 0; k < size[2]; k ++) {
        zv->SetValue(k, z[k]);
    }
    grid->SetZCoordinates(zv);

    /** read vars */
    for (Int v = 0; v < var_count; v ++) {
        vtkNew<vtkFloatArray> varv;
        varv->SetNumberOfComponents(var_dim[v]);
        varv->SetNumberOfTuples(count);
        varv->SetName(var_name[v].c_str());

        vector<Real> var(count*var_dim[v]);
        ifs.read((char*)var.data(), count*var_dim[v]*sizeof(Real));
        for (Int i = 0; i < size[0]; i ++) {
        for (Int j = 0; j < size[1]; j ++) {
        for (Int k = 0; k < size[2]; k ++) {
            Int id = index(i, j, k, size);
            for (Int m = 0; m < var_dim[v]; m ++) {
                Int component_id = id*var_dim[v] + m;
                Int vtk_id = vtkindex(i, j, k, m, size, var_dim[v]);
                varv->SetValue(vtk_id, var[component_id]);
            }
        }}}

        grid->GetPointData()->AddArray(varv);
    }

    ifs.close();
}

int main(int argc, char *argv[]) {
    vtkNew<vtkRectilinearGrid> grid;
    vtkNew<vtkXMLRectilinearGridWriter> writer;
    string ifilename(argv[1]);
    string ofilename = ifilename + ".vtr";
    writer->SetFileName(ofilename.c_str());
    writer->SetCompressionLevel(1);
    writer->SetInputData(grid);

    cout << "read from " << ifilename << endl;
    read_binary(ifilename, grid);

    writer->Write();
    cout << "write to " << ofilename << endl;
}
