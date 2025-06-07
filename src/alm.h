#pragma once

#include <vector>
#include <cmath>
#include "json.hpp"
#include "mpi_type.h"
#include "util.h"
#include "euler_angle.h"

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
    Real ow;
    Real formula[4];
    Real rot_speed;
    Real torque;
    Real thrust;
};

static nlohmann::json build_ap_props(
    const nlohmann::json &wt_prop_json,
    Int wt_count, Int ap_per_blade
) {
    auto &blade_json = wt_prop_json["blade"];
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

    Int blade_per_wt = wt_prop_json["number of blades"];
    Int ap_per_wt = blade_per_wt*ap_per_blade;
    Int ap_count = wt_count * ap_per_wt;
    Real wt_r = wt_prop_json["R"];
    Real hub_r = wt_prop_json["hub R"];
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

static void build_wt_props(
    const nlohmann::json &wt_prop_json,
    const nlohmann::json &wt_array_json,
    WindTurbine *&wt_lst, Int &wt_count
) {
    wt_count = wt_array_json.size();
    wt_lst = new WindTurbine[wt_count];
    Real tower = wt_prop_json["tower"];
    Real overhang = wt_prop_json["overhang"];
    for (Int i = 0; i < wt_count; i ++) {
        auto &wt_json = wt_array_json[i];
        auto &wt = wt_lst[i];
        auto &base = wt_json["base"];
        wt.base[0] = base[0];
        wt.base[1] = base[1];
        wt.base[2] = base[2];

        wt.rot_speed = wt_json["rotation speed"];

        wt.rot_center[0] = - overhang;
        wt.rot_center[1] = 0;
        wt.rot_center[2] = tower;

        auto &angle = wt_json["angle"];
        for (Int j = 0; j < 3; j ++) {
            auto &a = angle[j];
            if (
                a.is_number() && a.get<Real>() != 0
            ||  a.is_object()
            ) {
                switch (j)
                {
                case 0:
                    wt.angle_type = EulerAngle::Roll;
                    break;
                case 1:
                    wt.angle_type = EulerAngle::Pitch;
                    break;
                case 2:
                    wt.angle_type = EulerAngle::Yaw;
                    break;
                default:
                    wt.angle_type = EulerAngle::None;
                    break;
                }

                if (a.is_number()) {
                    wt.formula[0] = 0;
                    wt.formula[1] = 0;
                    wt.formula[2] = 0;
                    wt.formula[3] = deg_to_rad(a);
                } else if (a.is_object()) {
                    wt.formula[0] = deg_to_rad(a["amp"]);
                    wt.formula[1] = 2*Pi/a["T"].get<Real>();
                    wt.formula[2] = deg_to_rad(a["phase"]);
                    wt.formula[3] = deg_to_rad(a["offset"]);
                }

                break;
            }
        }
    }
}

static void lookup_cdcl_table(
    Real *cd_tbl, Real *cl_tbl, Real *atk_lst, Int atk_count,
    Real atk, Real &cd, Real &cl
) {
    Int i = find_floor_index(atk_lst, atk, atk_count);
    if (i < 0) {
        cd = cd_tbl[0];
        cl = cl_tbl[0];
    } else if (i >= atk_count - 1) {
        cd = cd_tbl[atk_count - 1];
        cl = cl_tbl[atk_count - 1];
    } else {
        cd = linear_interpolate(
            cd_tbl[i], cd_tbl[i + 1],
            atk_lst[i], atk_lst[i + 1],
            atk
        );
        cl = linear_interpolate(
            cl_tbl[i], cl_tbl[i + 1],
            atk_lst[i], atk_lst[i + 1],
            atk
        );
    }
}

static void update_ap_position(WindTurbine *wt_lst, Int wt_count, ActuatorPoint *ap_lst, Int ap_count, Int blade_per_wt, Int ap_per_blade, Real t) {
// #pragma acc kernels loop independent \
// present(wt_lst[:wt_count])
    for (Int wtid = 0; wtid < wt_count; wtid ++) {
        auto &wt = wt_lst[wtid];
        Real A = wt.formula[0];
        Real B = wt.formula[1];
        Real C = wt.formula[2];
        Real D = wt.formula[3];
        wt.angle = A*sin(B*t + C) + D;
        wt.ow = A*B*cos(B*t + C);
    }
#pragma acc update device(wt_lst[:wt_count])

    Int ap_per_wt = ap_per_blade*blade_per_wt;
#pragma acc kernels loop independent \
present(wt_lst[:wt_count], ap_lst[:ap_count])
    for (Int apid = 0; apid < ap_count; apid ++) {
        Int wtid = apid/ap_per_wt;
        Int bid = (apid%ap_per_wt)/ap_per_blade;
        auto &ap = ap_lst[apid];
        auto &wt = wt_lst[wtid];
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
    WindTurbine *wt_lst, Int wt_count,
    ActuatorPoint *ap_lst, Int ap_count, Int blade_per_wt, Int ap_per_blade,
    Real *cd_tbl[], Real *cl_tbl[], Real *atk_lst, Int atk_count,
    Real t, Int size[3], Int gc,
    MpiInfo *mpi
) {
    Int len = size[0]*size[1]*size[2];
    Int ap_per_wt = ap_per_blade*blade_per_wt;
    std::vector<MPI_Request> apreq(ap_count);
#pragma acc kernels loop independent \
present(wt_lst[:wt_count], ap_lst[:ap_count]) \
present(atk_lst[:atk_count], cd_tbl[:ap_count][:atk_count], cl_tbl[:ap_count][:atk_count]) \
present(U[:len]) \
present(x[:size[0]], y[:size[1]], z[:size[2]]) \
copyin(size[:3])
    for (Int apid = 0; apid < ap_count; apid ++) {
        Int wtid = apid/ap_per_wt;
        auto &ap = ap_lst[apid];
        auto &wt = wt_lst[wtid];
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

            Real Uap[3];

#pragma acc loop independent
            for (Int m = 0; m < 3; m ++) {
                Real u0 = U[id0][m];
                Real u1 = U[id1][m];
                Real u2 = U[id2][m];
                Real u3 = U[id3][m];
                Real u4 = U[id4][m];
                Real u5 = U[id5][m];
                Real u6 = U[id6][m];
                Real u7 = U[id7][m];
                Uap[m] = trilinear_interpolate(
                    u0, u1, u2, u3, u4, u5, u6, u7,
                    x0, x1, y0, y1, z0, z1, xp, yp, zp
                );
            }

            Real apxyz_rel[] = {
                ap.xyz[0] - wt.base[0],
                ap.xyz[1] - wt.base[1],
                ap.xyz[2] - wt.base[2]
            };
            Real Uap_[3];
            frame_transform_U(apxyz_rel, Uap, Uap_, wt.angle, wt.ow, wt.angle_type);

            Real ux_ = Uap_[0];
            Real ut_ = ap.r*wt.rot_speed + Uap_[1]*sin(ap.theta) - Uap_[2]*cos(ap.theta);
            Real Urel2 = ux_*ux_ + ut_*ut_;
            Real phi = atan(ux_/ut_);
            Real atk = rad_to_deg(phi) - ap.twist;
            Real cd, cl;
            lookup_cdcl_table(
                cd_tbl[apid], cl_tbl[apid], atk_lst, atk_count,
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
            // printf("%ld %lf %lf %lf\n", apid, ap.f[0], ap.f[1], ap.f[2]);
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
#pragma acc host_data use_device(ap_lst)
            MPI_Iallreduce(MPI_IN_PLACE, ap_lst[apid].f, 3, get_mpi_datatype<Real>(), MPI_SUM, MPI_COMM_WORLD, &apreq[apid]);
        }
    }

#pragma acc update host(ap_lst[:ap_count])
    std::vector<MPI_Request> torque_req(wt_count), thrust_req(wt_count);
// #pragma acc kernels loop independent \
// present(wt_lst[:wt_count], ap_lst[:ap_count])
    for (Int wtid = 0; wtid < wt_count; wtid ++) {
        auto &wt = wt_lst[wtid];
        Real torque = 0, thrust = 0;
// #pragma acc loop seq
        for (Int apid = wtid*ap_per_wt; apid < (wtid + 1)*ap_per_wt; apid ++) {
            torque += ap_lst[apid].torque;
            thrust += ap_lst[apid].thrust;
        }
        wt.torque = torque;
        wt.thrust = thrust;
    }

    if (mpi->size > 1) {
        for (Int wtid = 0; wtid < wt_count; wtid ++) {
// #pragma acc host_data use_device(wt_lst)
            MPI_Iallreduce(MPI_IN_PLACE, &wt_lst[wtid].torque, 1, get_mpi_datatype<Real>(), MPI_SUM, MPI_COMM_WORLD, &torque_req[wtid]);
// #pragma acc host_data use_device(wt_lst)
            MPI_Iallreduce(MPI_IN_PLACE, &wt_lst[wtid].thrust, 1, get_mpi_datatype<Real>(), MPI_SUM, MPI_COMM_WORLD, &thrust_req[wtid]);
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
    ActuatorPoint *ap_lst, Int ap_count, Real projection_width,
    Real x[], Real y[], Real z[],
    Int size[3], Int gc
) {
    Int len = size[0]*size[1]*size[2];
#pragma acc kernels loop independent collapse(3) \
present(f[:len], ap_lst[:ap_count]) \
present(x[:size[0]], y[:size[1]], z[:size[2]]) \
copyin(size[:3])
    for (Int i = gc; i < size[0] - gc; i ++) {
    for (Int j = gc; j < size[1] - gc; j ++) {
    for (Int k = gc; k < size[2] - gc; k ++) {
        Int id = index(i, j, k, size);
        Real fx = 0, fy = 0, fz = 0;
        Real xc = x[i], yc = y[j], zc = z[k];
        Real e = 1/cubic(projection_width*sqrt(Pi));
#pragma acc loop seq
        for (Int apid = 0; apid < ap_count; apid ++) {
            auto &ap = ap_lst[apid];
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

static void run_alm(
    Real U[][3], Real f[][3], Real projection_width,
    Real x[], Real y[], Real z[],
    WindTurbine *wt_lst, Int wt_count,
    ActuatorPoint *ap_lst, Int ap_count, Int blade_per_wt, Int ap_per_blade,
    Real *cd_tbl[], Real *cl_tbl[], Real *atk_lst, Int atk_count,
    Real t, Int size[3], Int gc,
    MpiInfo *mpi
) {
    if (ap_count <= 0) {
        return;
    }
    update_ap_position(
        wt_lst, wt_count,
        ap_lst, ap_count, blade_per_wt, ap_per_blade, t
    );
    calc_ap_force(
        U, x, y, z,
        wt_lst, wt_count,
        ap_lst, ap_count, blade_per_wt, ap_per_blade,
        cd_tbl, cl_tbl, atk_lst, atk_count,
        t, size, gc, mpi
    );
    distribute_ap_force(
        f,
        ap_lst, ap_count, projection_width,
        x, y, z,
        size, gc
    );
}