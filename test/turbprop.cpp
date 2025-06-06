#include <fstream>
#include <iostream>
#include <cstdio>
#include "../src/alm.h"

using namespace std;
using json = nlohmann::json;

int main() {
    ifstream wt_prop_ifs("turbine.json");
    auto &&wt_prop_json = json::parse(wt_prop_ifs);
    ifstream wt_lst_ifs("turbine_list.json");
    auto &&wt_lst_json = json::parse(wt_lst_ifs);

    WindTurbine *wt_lst;
    Int wt_count;
    build_wt_props(wt_prop_json, wt_lst_json, wt_lst, wt_count);

    printf("wt count = %ld\n", wt_count);
    for (Int i = 0; i < wt_count; i ++) {
        auto &wt = wt_lst[i];
        printf("wt %ld\n", i);
        printf("\tbase = (%lf %lf %lf)\n", wt.base[0], wt.base[1], wt.base[2]);
        printf("\trotation speed = %lf\n", wt.rot_speed);
        printf("\trotation center = (%lf %lf %lf)\n", wt.rot_center[0], wt.rot_center[1], wt.rot_center[2]);
        printf("\tangle type = %s\n", euler_angle_to_str(wt.angle_type).c_str());
        printf(
            "\tangle formula = %lf sin(%lf t + %lf) + %lf\n",
            wt.formula[0], wt.formula[1], wt.formula[2], wt.formula[3]
        );
    }

    delete[] wt_lst;
}