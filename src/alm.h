#pragma once

#include <vector>
#include <cmath>
#include "json.hpp"
#include "mpi_type.h"
#include "util.h"
#include "euler_angle.h"

struct AirFoil {
    Real r;
    Real chord;
    Real twist;
    std::vector<Real> cl_table;
    std::vector<Real> cd_table;
};

struct ActuatorPoint {
    Real U[3];
    Real xyz[3];
    Real r;
    Real chord;
    Real twist;
    Real theta;
    Real dr;
    Real f[3];
    Real torque;
    Real thrust;
};

struct WindTurbine {
    Real base[3];
    Real rot_center[3];
    EulerAngle angle_type;
    Real angle;
    Real angle_dt;
    Real formula[4];
    Real rot_speed;
    Real torque;
    Real thrust;
};

static nlohmann::json build_ap_props(
    const nlohmann::json &wt_json,
    Int wt_count, Int ap_per_blade
) {
    auto &blade_json = wt_json["blade"];
    auto &af_json = blade_json["airfoil"];
    auto &atk_json = blade_json["attack angle"];

    Int af_count = af_json.size();
    Int atk_count = atk_json.size();

    std::vector<Real> afr;
    std::vector<Real> aftwist;
    std::vector<Real> afchord;
    std::vector<std::vector<Real>> afcd;
    std::vector<std::vector<Real>> afcl;
    for (auto &af : af_json) {
        afr.push_back(af["r/R"]);
        aftwist.push_back(af["twist[deg]"]);
        afchord.push_back(af["chord/R"]);
        afcd.push_back(af["Cd"].get<std::vector<Real>>());
        afcl.push_back(af["Cl"].get<std::vector<Real>>());
    }

    Int blade_per_wt = wt_json["number of blades"];
    Int ap_per_wt = blade_per_wt*ap_per_blade;
    Int ap_count = wt_count * ap_per_wt;
    Real wt_r = wt_json["R"];
    Real hub_r = wt_json["hub R"];
    Real dr = (wt_r - hub_r)/ap_per_blade;
    auto aps = nlohmann::json::array();
    for (Int i = 0; i < ap_count; i ++) {
        Real r = hub_r + (i % ap_per_blade + 0.5)*dr;
        nlohmann::json ap;
        ap["r"] = r;
        ap["dr"] = dr;

        Int af = find_floor_index(afr.data(), r, af_count);
        if (af < 0) {
            ap["chord"] = afchord[0];
            ap["twist"] = aftwist[0];
            ap["cd"] = afcd[0];
            ap["cl"] = afcl[0];
        } else if (af >= af_count - 1) {
            ap["chord"] = afchord[af_count - 1];
            ap["twist"] = aftwist[af_count - 1];
            ap["cd"] = afcd[af_count - 1];
            ap["cl"] = afcl[af_count - 1];
        } else {
            Real a1 = (r - afr[af])/(afr[af + 1] - afr[af]);
            Real a0 = 1 - a1;
            ap["chord"] = a0*afchord[af] + a1*afchord[af + 1];
            ap["twist"] = a0*aftwist[af] + a1*aftwist[af + 1];
            std::vector<Real> cd(atk_count), cl(atk_count);
            for (Int j = 0; j < atk_count; j ++) {
                cd[j] = a0*afcd[af][j] + a1*afcd[af + 1][j];
                cl[j] = a0*afcl[af][j] + a1*afcl[af + 1][j];
            }
            ap["cd"] = cd;
            ap["cl"] = cl;
        }
        aps.push_back(ap);
    }

    nlohmann::json tmp;
    tmp["ap"] = aps;
    tmp["attack"] = atk_json;
    return tmp;
}

static void build_ap_props(
    const nlohmann::json &ap_prop,
    ActuatorPoint *&ap_lst, Real *&atk_lst, Real **&cd_tbl, Real **&cl_tbl,
    Int &ap_count, Int &atk_count
) {
    auto &ap_json_lst = ap_prop["ap"];
    auto &atk_json_lst = ap_prop["attack"];
    ap_count = ap_json_lst.size();
    atk_count = atk_json_lst.size();

    atk_lst = new Real[atk_count];
    for (Int j = 0; j < atk_count; j ++) {
        atk_lst[j] = atk_json_lst[j];
    }

    ap_lst = new ActuatorPoint[ap_count];
    cd_tbl = new Real*[ap_count];
    cl_tbl = new Real*[ap_count];

    for (Int i = 0; i < ap_count; i ++) {
        auto &ap_json = ap_json_lst[i];
        auto &ap = ap_lst[i];
        ap.chord = ap_json["chord"];
        ap.twist = ap_json["twist"];
        ap.r = ap_json["r"];
        ap.dr = ap_json["dr"];
        cd_tbl[i] = new Real[atk_count];
        auto &cd_json = ap_json["cd"];
        for (Int j = 0; j < atk_count; j ++) {
            cd_tbl[i][j] = cd_json[j];
        }
        cl_tbl[i] = new Real[atk_count];
        auto &cl_json = ap_json["cl"];
        for (Int j = 0; j < atk_count; j ++) {
            cl_tbl[i][j] = cl_json[j];
        }
    }
}

static void lookup_cdcl_table(
    Real *cd_table, Real *cl_table, Real *atk_table, Int atk_count,
    Real atk, Real &cd, Real &cl
) {
    Int i = find_floor_index(atk_table, atk, atk_count);
    if (i < 0) {
        cd = cd_table[0];
        cl = cl_table[0];
    } else if (i >= atk_count - 1) {
        cd = cd_table[atk_count - 1];
        cl = cl_table[atk_count - 1];
    } else {
        cd = linear_interpolate(
            cd_table[i], cd_table[i + 1],
            atk_table[i], atk_table[i + 1],
            atk
        );
        cl = linear_interpolate(
            cl_table[i], cl_table[i + 1],
            atk_table[i], atk_table[i + 1],
            atk
        );
    }
}

static void update_ap_position(WindTurbine *wt_list, Int wt_count, ActuatorPoint *ap_list, Int ap_count, Int blade_per_wt, Int ap_per_blade, Real t) {
    for (Int wtid = 0; wtid < wt_count; wtid ++) {
        auto &wt = wt_list[wtid];
        Real A = wt.formula[0];
        Real B = wt.formula[1];
        Real C = wt.formula[2];
        Real D = wt.formula[3];
        wt.angle = A*sin(B*t + C) + D;
        wt.angle_dt = A*B*cos(B*t + C);
    }

    Int ap_per_wt = ap_per_blade*blade_per_wt;
    for (Int apid = 0; apid < ap_count; apid ++) {
        Int wtid = apid/ap_per_wt;
        Int bid = (apid%ap_per_wt)/ap_per_blade;
        auto &ap = ap_list[apid];
        auto &wt = wt_list[wtid];
        Real start_theta = bid*(2*Pi/blade_per_wt);
        ap.theta = start_theta + t*wt.rot_speed;
        Real xyz[] = {
            wt.rot_center[0],
            wt.rot_center[1] + ap.r*cos(ap.theta),
            wt.rot_center[2] + ap.r*sin(ap.theta)
        };
        frame_transform(xyz, ap.xyz, - wt.angle, wt.angle_type);
        ap.xyz[0] += wt.base[0];
        ap.xyz[1] += wt.base[1];
        ap.xyz[2] += wt.base[2];
    }
}

static void calc_ap_force(
    Real U[][3],
    Real x[], Real y[], Real z[],
    WindTurbine *wt_list, Int wt_count,
    ActuatorPoint *ap_list, Int ap_count, Int blade_per_wt, Int ap_per_blade,
    Real *cd_table[], Real *cl_table[], Real *atk_table, Int atk_count,
    Real t, Int size[3], Int gc,
    MpiInfo *mpi
) {
    Int ap_per_wt = ap_per_blade*blade_per_wt;
    std::vector<MPI_Request> apreq(ap_count);
    for (Int apid = 0; apid < ap_count; apid ++) {
        Int wtid = apid/ap_per_wt;
        auto &ap = ap_list[apid];
        auto &wt = wt_list[wtid];
        Int i = find_floor_index(x, ap.xyz[0], size[0]);
        Int j = find_floor_index(y, ap.xyz[1], size[1]);
        Int k = find_floor_index(z, ap.xyz[2], size[2]);

        if (
            i >= gc && i < size[0] - gc
        &&  j >= gc && j < size[1] - gc
        &&  k >= gc && k < size[2] - gc
        ) {
            Int id0 = index(i    , j    , k    , size);
            Int id1 = index(i + 1, j    , k    , size);
            Int id2 = index(i    , j + 1, k    , size);
            Int id3 = index(i + 1, j + 1, k    , size);
            Int id4 = index(i    , j    , k + 1, size);
            Int id5 = index(i + 1, j    , k + 1, size);
            Int id6 = index(i    , j + 1, k + 1, size);
            Int id7 = index(i + 1, j + 1, k + 1, size);

            Real x0 = x[i], x1 = x[i + 1];
            Real y0 = y[j], y1 = y[j + 1];
            Real z0 = z[k], z1 = z[k + 1];
            Real xp = ap.xyz[0], yp = ap.xyz[1], zp = ap.xyz[2];

            Real u0 = U[id0][0];
            Real u1 = U[id1][0];
            Real u2 = U[id2][0];
            Real u3 = U[id3][0];
            Real u4 = U[id4][0];
            Real u5 = U[id5][0];
            Real u6 = U[id6][0];
            Real u7 = U[id7][0];
            Real uap = trilinear_interpolate(
                u0, u1, u2, u3, u4, u5, u6, u7,
                x0, x1, y0, y1, z0, z1, xp, yp, zp
            );

            Real v0 = U[id0][1];
            Real v1 = U[id1][1];
            Real v2 = U[id2][1];
            Real v3 = U[id3][1];
            Real v4 = U[id4][1];
            Real v5 = U[id5][1];
            Real v6 = U[id6][1];
            Real v7 = U[id7][1];
            Real vap = trilinear_interpolate(
                v0, v1, v2, v3, v4, v5, v6, v7,
                x0, x1, y0, y1, z0, z1, xp, yp, zp
            );

            Real w0 = U[id0][2];
            Real w1 = U[id1][2];
            Real w2 = U[id2][2];
            Real w3 = U[id3][2];
            Real w4 = U[id4][2];
            Real w5 = U[id5][2];
            Real w6 = U[id6][2];
            Real w7 = U[id7][2];
            Real wap = trilinear_interpolate(
                w0, w1, w2, w3, w4, w5, w6, w7,
                x0, x1, y0, y1, z0, z1, xp, yp, zp
            );

            Real apxyz_[] = {
                ap.xyz[0] - wt.base[0],
                ap.xyz[1] - wt.base[1],
                ap.xyz[2] - wt.base[2]
            };
            Real Uap[] = {uap, vap, wap}, Uap_[3];
            frame_transform_dt(apxyz_, Uap, Uap_, wt.angle, wt.angle_dt, wt.angle_type);

            Real ux_ = Uap_[0];
            Real ut_ = ap.r*wt.rot_speed + Uap_[1]*sin(ap.theta) - Uap_[2]*cos(ap.theta);
            Real Urel2 = ux_*ux_ + ut_*ut_;
            Real phi = atan(ux_/ut_);
            Real atk = phi - ap.twist;
            Real cd, cl;
            lookup_cdcl_table(
                cd_table[apid], cl_table[apid], atk_table, atk_count,
                atk, cd, cl
            );
            Real fd = 0.5*cd*Urel2*ap.chord*ap.dr;
            Real fl = 0.5*cl*Urel2*ap.chord*ap.dr;
            Real fx_ = fl*cos(phi) + fd*sin(phi);
            Real ft_ = fl*sin(phi) - fd*cos(phi);
            ft_ *= sign(wt.rot_speed);
            Real f_[] = {
              - fx_,
                ft_*sin(ap.theta),
              - ft_*cos(ap.theta)
            };
            frame_transform(f_, ap.f, - wt.angle, wt.angle_type);
            ap.torque = fabs(ft_)*ap.r;
            ap.thrust = fx_;
        } else {
            ap.f[0] = 0;
            ap.f[1] = 0;
            ap.f[2] = 0;
            ap.torque = 0;
            ap.thrust = 0;
        }
    }
    
    if (mpi->size > 1) {
        for (Int apid = 0; apid < ap_count; apid ++) {
            MPI_Iallreduce(MPI_IN_PLACE, ap_list[apid].f, 3, get_mpi_datatype<Real>(), MPI_SUM, MPI_COMM_WORLD, &apreq[apid]);
        }
    }

    std::vector<MPI_Request> torque_req(wt_count), thrust_req(wt_count);
    for (Int wtid = 0; wtid < wt_count; wtid ++) {
        auto &wt = wt_list[wtid];
        Real torque = 0, thrust = 0;
        for (Int apid = wtid*ap_per_wt; apid < (wtid + 1)*ap_per_wt; apid ++) {
            torque += ap_list[apid].torque;
            thrust += ap_list[apid].thrust;
        }
        wt.torque = torque;
        wt.thrust = thrust;
    }

    if (mpi->size > 1) {
        for (Int wtid = 0; wtid < wt_count; wtid ++) {
            MPI_Iallreduce(MPI_IN_PLACE, &wt_list[wtid].torque, 1, get_mpi_datatype<Real>(), MPI_SUM, MPI_COMM_WORLD, &torque_req[wtid]);
            MPI_Iallreduce(MPI_IN_PLACE, &wt_list[wtid].thrust, 1, get_mpi_datatype<Real>(), MPI_SUM, MPI_COMM_WORLD, &thrust_req[wtid]);
        }
    }

    if (mpi->size > 1) {
        MPI_Waitall(ap_count, apreq.data(), MPI_STATUSES_IGNORE);
        MPI_Waitall(wt_count, torque_req.data(), MPI_STATUSES_IGNORE);
        MPI_Waitall(wt_count, thrust_req.data(), MPI_STATUSES_IGNORE);
    }
}

static void distribute_ap_force(
    Real f[][3],
    ActuatorPoint *ap_list, Int ap_count, Real projection_width,
    Real x[], Real y[], Real z[],
    Int size[3], Int gc
) {
    for (Int i = gc; i < size[0] - gc; i ++) {
    for (Int j = gc; j < size[1] - gc; j ++) {
    for (Int k = gc; k < size[2] - gc; k ++) {
        Int id = index(i, j, k, size);
        Real fx = 0, fy = 0, fz = 0;
        Real xc = x[i], yc = y[j], zc = z[k];
        Real e = 1/cubic(projection_width*sqrt(Pi));
        for (Int apid = 0; apid < ap_count; apid ++) {
            auto &ap = ap_list[apid];
            Real r2 = square(xc - ap.xyz[0]) + square(yc - ap.xyz[1]) + square(zc - ap.xyz[2]);
            Real wght = e*exp(- r2/square(projection_width));

            fx += wght*ap.f[0];
            fy += wght*ap.f[1];
            fz += wght*ap.f[2];
        }
        f[id][0] = fx;
        f[id][1] = fy;
        f[id][2] = fz;
    }}}
}