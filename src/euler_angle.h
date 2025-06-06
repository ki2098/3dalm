#pragma once

#include <cmath>
#include <string>
#include "type.h"

enum class EulerAngle {Roll, Pitch, Yaw, Undefined};

static std::string euler_angle_to_str(EulerAngle type) {
    switch (type) {
    case EulerAngle::Roll:
        return "roll"; 
    case EulerAngle::Pitch:
        return "pitch"; 
    case EulerAngle::Yaw:
        return "yaw"; 
    default:
        return "undefined"; 
    }
}

static EulerAngle str_2_euler_angle(const std::string &str) {
    if (str == "roll") {
        return EulerAngle::Roll;
    } else if (str == "pitch") {
        return EulerAngle::Pitch;
    } else if (str == "yaw") {
        return EulerAngle::Yaw;
    } else {
        return EulerAngle::Undefined;
    }
}

static void frame_transform(Real xyz[3], Real xyz_[3], Real angle, EulerAngle type) {
    Real c = cos(angle);
    Real s = sin(angle);
    Real x = xyz[0];
    Real y = xyz[1];
    Real z = xyz[2];
    switch (type) {
    case EulerAngle::Roll:
        xyz_[0] = x;
        xyz_[1] =   c*y + s*z;
        xyz_[2] = - s*y + c*z;
        break;
    case EulerAngle::Pitch:
        xyz_[0] =   c*x - s*z;
        xyz_[1] = y;
        xyz_[2] = - s*x + c*z;
        break;
    case EulerAngle::Yaw:
        xyz_[0] =   c*x + s*y;
        xyz_[1] = - s*x + c*y;
        xyz_[2] = z;
        break;
    default:
        xyz_[0] = x;
        xyz_[1] = y;
        xyz_[2] = z;
        break;
    }
}

static void frame_transform_dt(Real xyz[3], Real U[3], Real U_[3], Real angle, Real ow, EulerAngle type) {
    Real c = cos(angle);
    Real s = sin(angle);
    Real x = xyz[0];
    Real y = xyz[1];
    Real z = xyz[2];
    Real u = U[0];
    Real v = U[1];
    Real w = U[2];
    switch (type) {
    case EulerAngle::Roll: {
        Real dRdtx[] = {
            0.,
            ow*(- s*y + c*z),
            ow*(- c*y - s*z)
        };
        Real Rdtdx[] = {
            u,
            c*v + s*w,
          - s*v + v*w
        };
        U_[0] = dRdtx[0] + Rdtdx[0];
        U_[1] = dRdtx[1] + Rdtdx[1];
        U_[2] = dRdtx[2] + Rdtdx[2];
        break;
    }
    case EulerAngle::Pitch: {
        Real dRdtx[] = {
            ow*(- s*x - c*z),
            0.,
            ow*(  c*x - s*z)
        };
        Real Rdtdx[] = {
            c*u - s*w,
            v,
            s*u + c*w
        };
        U_[0] = dRdtx[0] + Rdtdx[0];
        U_[1] = dRdtx[1] + Rdtdx[1];
        U_[2] = dRdtx[2] + Rdtdx[2];
        break;
    }
    case EulerAngle::Yaw: {
        Real dRdtx[] = {
            ow*(- s*x + c*y),
            ow*(- c*s - s*y),
            0.
        };
        Real Rdtdx[] = {
            c*u + s*v,
          - s*u + c*v,
            w
        };
        U_[0] = dRdtx[0] + Rdtdx[0];
        U_[1] = dRdtx[1] + Rdtdx[1];
        U_[2] = dRdtx[2] + Rdtdx[2];
        break;
    }
    default: {
        U_[0] = u;
        U_[1] = v;
        U_[2] = w;
        break;
    }}
}