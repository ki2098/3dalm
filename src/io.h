#pragma acc

#include <fstream>

/**
 * header format:
 * int      sz[0]
 * int      sz[1]
 * int      sz[2]
 * int      gc
 * int      vn
 * int      vd[0]
 * ...
 * int      vd[vn-1]
 * int      step
 * double   time
 */
static void write_binary_file(
    std::string path,
    int vn,
    double *v[],
    int vd[],
    int step,
    double time,
    int sz[3],
    int gc
) {
    std::ofstream out(path, std::ios::binary);
    out.write((char*)sz, sizeof(int)*3);
    out.write((char*)&gc, sizeof(int));
    out.write((char*)&vn, sizeof(int));
    out.write((char*)vd, sizeof(int)*vn);
    out.write((char*)&step, sizeof(int));
    out.write((char*)&time, sizeof(double));
    for (int n = 0; n < vn; n ++) {
        out.write((char*)&v[n], sz[0]*sz[1]*sz[2]*vd[n]*sizeof(double));
    }
}

static void write_csv_file(
    std::string path,
    int vn,
    double *v[],
    int *vd,
    std::string vname[],
    double x[],
    double y[],
    double z[],
    int sz[3],
    int gc
) {
    std::ofstream out(path);
    out << "x,y,z";
    for (int n = 0; n < vn; n ++) {
        for (int m = 0; m < vd[n]; m ++) {
            if (vd[n] > 1) {
                out << "," << (vname[n] + ":" + std::to_string(m));
            } else {
                out << "," << vname[n];
            }
            
        }
    }
    out << std::endl;
    for (int k = 0; k < sz[2]; k ++) {
        for (int j = 0; j < sz[1]; j ++) {
            for (int i = 0; i < sz[0]; i ++) {
                out << x[i] << "," << y[j] << "," << z[k];
                for (int n = 0; n < vn; n ++) {
                    for (int m = 0; m < vd[n]; m ++) {
                        out << "," << v[n][i*sz[1]*sz[2]*vd[n] + j*sz[2]*vd[n] + k*vd[n] + m];
                    }
                }
                out << std::endl;
            }
        }
    }
}