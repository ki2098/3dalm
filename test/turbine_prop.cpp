#include <fstream>
#include <iostream>
#include <cstdio>
#include "../src/alm.h"

using namespace std;
using json = nlohmann::json;

int main() {
    ifstream ifs("turbine.json");
    auto turbine_prop = json::parse(ifs);
    auto && ap_prop = build_ap_props(turbine_prop, 2, 20);
    ActuatorPoint *ap_lst;
    Real *atk_lst, **cd_tbl, **cl_tbl;
    Int ap_count, atk_count;
    build_ap_props(ap_prop, ap_lst, atk_lst, cd_tbl, cl_tbl, ap_count, atk_count);

    printf("ap count = %ld, atk acount = %ld\n", ap_count, atk_count);
    for (Int i = 0; i < ap_count; i ++) {
        printf("ap %ld\n", i);
        auto &ap = ap_lst[i];
        printf("\tr = %lf\n", ap.r);
        printf("\tdr = %lf\n", ap.dr);
        printf("\tchord = %lf\n", ap.chord);
        printf("\ttwist = %lf\n", ap.twist);
        printf("\tcd = \n");
        for (Int j = 0; j < atk_count; j ++) {
            printf("\t\t%lf\n", cd_tbl[i][j]);
        }
        printf("\tcl = \n");
        for (Int j = 0; j < atk_count; j ++) {
            printf("\t\t%lf\n", cl_tbl[i][j]);
        }
    }
    printf("atk\n");
    for (Int j = 0; j < atk_count; j ++) {
        printf("\t%lf\n", atk_lst[j]);
    }
}