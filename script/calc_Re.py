#!/usr/bin/env python

T0 = float(input('T0 [Celcius] = '))
mu0 = float(input('mu0 [Pa*s] = '))
S = float(input('S = '))
T = float(input('T [Celcius] = '))
rho = float(input('rho [kg/m^3] = '))
U0 = float(input('U0 [m/s] = '))
L0 = float(input('L0 [m] = '))

T_offset = 273.15

T0 += T_offset
T += T_offset

mu = mu0 * (T/T0)**(3/2) * (T0 + S)/(T + S)
nu = mu/rho
Re = L0*U0/nu

print('mu =', mu, '[Pa*s]')
print('nu =', nu, '[m^2/s]')
print('Re =', Re)