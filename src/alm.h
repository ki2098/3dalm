#pragma once

#include <vector>
#include <cmath>
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
    Real angle_dt;
    Real formula[4];
    Real rot_speed;
    Real torque;
    Real thrust;
};

static void lookup_cdcl_table(
    Real *cd_table, Real *cl_table, Real *attack_table, Int attack_count,
    Real attack, Real &cd, Real &cl
) {
    Int i = find_floor_index(attack_table, attack, attack_count);
    if (i < 0) {
        cd = cd_table[0];
        cl = cl_table[0];
    } else if (i >= attack_count - 1) {
        cd = cd_table[attack_count - 1];
        cl = cl_table[attack_count - 1];
    } else {
        cd = linear_interpolate(
            cd_table[i], cd_table[i + 1],
            attack_table[i], attack_table[i + 1],
            attack
        );
        cl = linear_interpolate(
            cl_table[i], cl_table[i + 1],
            attack_table[i], attack_table[i + 1],
            attack
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
    Real *cd_table[], Real *cl_table[], Real *attack_table, Int attack_count,
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
            Real attack = phi - ap.twist;
            Real cd, cl;
            lookup_cdcl_table(
                cd_table[apid], cl_table[apid], attack_table, attack_count,
                attack, cd, cl
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
        for (Int apid = 0; apid < ap_count; apid ++) {
            auto &ap = ap_list[apid];
            Real xc = x[i], yc = y[j], zc = z[k];
            Real r2 = square(xc - ap.xyz[0]) + square(yc - ap.xyz[1]) + square(zc - ap.xyz[2]);
            Real wght = (1/cubic(projection_width*sqrt(Pi)))*exp(- r2/square(projection_width));

            fx += wght*ap.f[0];
            fy += wght*ap.f[1];
            fz += wght*ap.f[2];
        }
        f[id][0] = fx;
        f[id][1] = fy;
        f[id][2] = fz;
    }}}
}