#!/usr/bin/python
# -*- coding: utf-8 -*-
# vim: tabstop=4:softtabstop=4:shiftwidth=4:expandtab
"""
libfitorbit: an intuitive interface for fitorbit-like functionality

"""


from __future__ import print_function
from __future__ import division

import numpy as np
import os
import ephem        # pip install pyephem
import scipy.interpolate as si
import scipy.linalg as sl
import scipy.optimize as so

import fitorbit_parfile as parfile

try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

# Use long doubles for the calculations (from fitorbit)
SECS_PER_DAY    = np.float128('86400.0')
DEG2RAD         = np.float128('1.7453292519943295769236907684886127134428718885417e-2')
RAD2DEG         = np.float128('57.295779513082320876798154814105170332405472466564')
C               = np.float128('2.99792458e8')
SUNMASS         = np.float128('4.925490947e-6')


def pardict_to_array(pardict, which='BT'):
    """
    Given a dictionary of parameter values, return an array with the parameters
    in the right order (so it can be used for the binary models)

    @param pardict:     Dictionary of orbit model parameters
    @param which:       Which binary model we are arranging for

    @return x:          Array of model parameters
    """
    x = None
    if which=='BT' or which=='BTG':
        x = np.zeros(10, dtype=np.float128)
        if 'P0' in pardict:
            x[0] = np.float128(pardict['P0'].val)

        if 'P1' in pardict:
            x[1] = np.float128(pardict['P1'].val)

        if 'PEPOCH' in pardict:
            x[2] = np.float128(pardict['PEPOCH'].val)

        if 'PB' in pardict:
            x[3] = np.float128(pardict['PB'].val)

        if 'ECC' in pardict:
            x[4] = np.float128(pardict['ECC'].val)

        if 'A1' in pardict:
            x[5] = np.float128(pardict['A1'].val)

        if 'T0' in pardict:
            x[6] = np.float128(pardict['T0'].val)

        if 'OM' in pardict:
            x[7] = np.float128(pardict['OM'].val)

        if 'RA' in pardict:
            if isinstance(pardict['RA'].val, basestring):
                x[8] = np.float128(ephem.hours(str(pardict['RA'].val)))
            else:
                x[8] = np.float128(pardict['RA'].val)

        if 'DEC' in pardict:
            if isinstance(pardict['DEC'].val, basestring):
                x[9] = np.float128(ephem.degrees(str(pardict['DEC'].val)))
            else:
                x[9] = np.float128(pardict['DEC'].val)
    elif which=='RED':
        x = np.zeros(6, dtype=np.float128)
        if 'P0' in pardict:
            x[0] = np.float128(pardict['P0'].val)

        if 'RAMP' in pardict:
            x[1] = np.float128(pardict['RAMP'].val)

        if 'OM' in pardict:
            x[2] = np.float128(pardict['OM'].val)

        if 'T0' in pardict:
            x[3] = np.float128(pardict['T0'].val)

        if 'ECC' in pardict:
            x[4] = np.float128(pardict['ECC'].val)

        if 'PB' in pardict:
            x[5] = np.float128(pardict['PB'].val)
    else:
        raise NotImplemented("Only BT works for now")

    return x

def array_to_pardict(parameters, which='BT'):
    """
    Given an array of parameter values, return an odereddict with the parameter
    values

    @param parameters:  Array of orbit model parameters
    @param which:       Which binary model we are arranging for

    @return pardict:    Parameter dictionary with parameter values
    """
    pardict = OrderedDict()
    if which=='BT' or which=='BTG':
        pardict['P0'] = createOrbitPar('P0')
        pardict['P0'].val = parameters[0]

        pardict['P1'] = createOrbitPar('P1')
        pardict['P1'].val = parameters[1]

        pardict['PEPOCH'] = createOrbitPar('PEPOCH')
        pardict['PEPOCH'].val = parameters[2]

        pardict['PB'] = createOrbitPar('PB')
        pardict['PB'].val = parameters[3]

        pardict['ECC'] = createOrbitPar('ECC')
        pardict['ECC'].val = parameters[4]

        pardict['A1'] = createOrbitPar('A1')
        pardict['A1'].val = parameters[5]

        pardict['T0'] = createOrbitPar('T0')
        pardict['T0'].val = parameters[6]

        pardict['OM'] = createOrbitPar('OM')
        pardict['OM'].val = parameters[7]

        pardict['RA'] = createOrbitPar('RA')
        pardict['RA'].val = parameters[8]

        pardict['DEC'] = createOrbitPar('DEC')
        pardict['DEC'].val = parameters[9]


        #pardict['XX'] = createOrbitPar('XX')
        #pardict['XX'].val = parameters[10]

        #pardict['XX'] = createOrbitPar('XX')
        #pardict['XX'].val = parameters[11]
    elif which=='RED':
        pardict['P0'] = createOrbitPar('P0')
        pardict['P0'].val = parameters[0]

        pardict['RAMP'] = createOrbitPar('RAMP')
        pardict['RAMP'].val = parameters[1]

        pardict['OM'] = createOrbitPar('OM')
        pardict['OM'].val = parameters[2]

        pardict['T0'] = createOrbitPar('T0')
        pardict['T0'].val = parameters[3]

        pardict['ECC'] = createOrbitPar('ECC')
        pardict['ECC'].val = parameters[4]

        pardict['PB'] = createOrbitPar('PB')
        pardict['PB'].val = parameters[5]
    elif which=='DD':
        pass

    return pardict

def RED2BT(redpars, btpars):
    """
    Convert the reduced model parameters to the BT model parameters
    """
    P0r = redpars[0]
    RAMP = redpars[1]
    OM = redpars[2]
    T0 = redpars[3]
    ECC = redpars[4]
    PB = redpars[5]

    # Do some conversion adjustments
    P0 = P0r - RAMP*ECC*np.cos(DEG2RAD*OM)
    A1 = RAMP * PB * np.sqrt(1-ECC**2) * SECS_PER_DAY / (2*np.pi*P0)

    btpars[0] = P0
    btpars[1] = 0.0
    btpars[3] = PB
    btpars[4] = ECC
    btpars[5] = A1
    btpars[6] = T0
    btpars[7] = OM

    return btpars


def eccentric_anomaly(E, mean_anomaly):
    """
    eccentric_anomaly(mean_anomaly):
            Return the eccentric anomaly in radians, given a set of mean_anomalies
            in radians (phase).
    """
    ma = np.fmod(mean_anomaly, 2*np.pi)
    ma = np.where(ma < 0.0, ma+2*np.pi, ma)
    eccentricity = E
    ecc_anom_old = ma
    #print ma
    ecc_anom = ma + eccentricity*np.sin(ecc_anom_old)
    iter = 0
    # This is a simple iteration to solve Kepler's Equation

    if (np.alen(ecc_anom) >1):
      while (np.maximum.reduce(np.fabs(ecc_anom-ecc_anom_old)) > 5e-15) and \
              iter < 50:
        ecc_anom_old = ecc_anom
        ecc_anom = ma + eccentricity*np.sin(ecc_anom_old)
        #print(">1",  ecc_anom, iter)
        iter+=1

    elif(np.alen(ecc_anom) ==1):
      while (np.fabs(ecc_anom-ecc_anom_old) > 5e-15) and \
              iter < 50:
        ecc_anom_old = ecc_anom
        ecc_anom = ma + eccentricity*np.sin(ecc_anom_old)
        #print("=1", ecc_anom, iter)
        iter+=1

    return ecc_anom


def eccentric_anomaly2(eccentricity, mean_anomaly):
    """
    eccentric_anomaly(mean_anomaly):
    Return the eccentric anomaly in radians, given a set of mean_anomalies
    in radians.
    """
    TWOPI = 2 * np.pi
    ma = np.fmod(mean_anomaly, TWOPI)
    ma = np.where(ma < 0.0, ma+TWOPI, ma)
    ecc_anom_old = ma
    ecc_anom = ma + eccentricity*np.sin(ecc_anom_old)

    # This is a simple iteration to solve Kepler's Equation
    while (np.maximum.reduce(np.fabs(ecc_anom-ecc_anom_old)) > 5e-15):
        ecc_anom_old = ecc_anom[:]
        ecc_anom = ma + eccentricity * np.sin(ecc_anom_old)

    return ecc_anom

def true_from_eccentric(e, ea):
    """Compute the true anomaly from the eccentric anomaly.

    Inputs:
        e - the eccentricity
        ea - the eccentric anomaly

    Outputs:
        true_anomaly - the true anomaly
        true_anomaly_de - derivative of true anomaly with respect to e
        true_anomaly_prime - derivative of true anomaly with respect to
            eccentric anomaly
    """
    true_anomaly = 2*np.arctan2(np.sqrt(1+e)*np.sin(ea/2),
                                np.sqrt(1-e)*np.cos(ea/2))
    true_anomaly_de = (np.sin(ea)/
            (np.sqrt(1-e**2)*(1-e*np.cos(ea))))
    true_anomaly_prime = (np.sqrt(1-e**2)/(1-e*np.cos(ea)))
    return true_anomaly, true_anomaly_de, true_anomaly_prime


def BT_delay(t, PB, T0, A1, OM, ECC=0.0, EDOT=0.0, PBDOT=0.0, XDOT=0.0, \
        OMDOT=0.0, GAMMA=0.0):
    """
    Delay due to pulsar binary motion. Model:
    Blandford & Teukolsky (1976), ApJ, 205, 580-591
    More precisely, equation (5) of:
    Taylor & Weisberg (1989), ApJ, 345, 434-450

    @param t:       Time series in MJDs
    @param PB:      Binary period
    @param T0:      Epoch of periastron passage
    @param A1:      Projected semi-major axis (a*sin(i)) on LOS (line-of-sight)
    @param OM:      Longitude of periastron (omega)
    @param ECC:     Eccentricity of the orbit [0.0]
    @param EDOT:    Time-derivative of ECC [0.0]
    @param PBDOT:   Time-derivative of PB [0.0]
    @param XDOT:    Time-derivative of a*sin(i)  [0.0]
    @param OMDOT:   Time-derivative of OMEGA [0.0]
    @param GAMMA:   Gravitational redshift and time dilation
    """
    tt0 = (t-np.float128(T0)) * SECS_PER_DAY 

    pb = PB * SECS_PER_DAY
    ecc = ECC + EDOT * tt0

    # TODO: Check this assertion. This mechanism is probably too strong
    # Probably better to create a BadParameter signal to raise,
    # catch it and give a new value to eccentricity?
    assert np.all(np.logical_and(ecc >= 0, ecc <= 1)), \
        "BT model: Eccentricity goes out of range"

    asini  = A1 + XDOT * tt0

    # XPBDOT exists in other models, not BT. In Tempo2 it is set to 0.
    # Check if it even makes sense to keep it here.
    xpbdot = 0.0

    omega = (OM + OMDOT*tt0/(SECS_PER_DAY*365.25)) * DEG2RAD
    pbdot = PBDOT

    orbits = tt0 / pb - 0.5 * (pbdot + xpbdot) * (tt0 / pb) ** 2
    norbits = np.array(np.floor(orbits), dtype=np.long)
    phase = 2 * np.pi * (orbits - norbits)
    bige = eccentric_anomaly(ecc, phase)

    tt = 1.0 - ecc ** 2
    som = np.sin(omega)
    com = np.cos(omega)

    alpha = asini * som
    beta = asini * com * np.sqrt(tt)
    sbe = np.sin(bige)
    cbe = np.cos(bige)
    q = alpha * (cbe - ecc) + (beta + GAMMA) * sbe
    r = -alpha * sbe + beta * cbe
    s = 1.0 / (1.0 - ecc * cbe)

    return -q + (2 * np.pi / pb) * q * r * s

    """
      if (param==param_pb)
        return -2.0*M_PI*r*s/pb*SECDAY*tt0/(SECDAY*pb) * SECDAY;  /* fctn(12+j) */
      else if (param==param_a1)
        return (som*(cbe-ecc) + com*sbe*sqrt(tt));                /* fctn(9+j) */
      else if (param==param_ecc)
        return -(alpha*(1.0+sbe*sbe-ecc*cbe)*tt - beta*(cbe-ecc)*sbe)*s/tt; /* fctn(10+j) */
      else if (param==param_om)
        return asini*(com*(cbe-ecc) - som*sqrt(tt)*sbe);          /* fctn(13+j) */
      else if (param==param_t0)
        return -2.0*M_PI/pb*r*s*SECDAY;                           /* fctn(11+j) */
      else if (param==param_pbdot)
        return 0.5*(-2.0*M_PI*r*s/pb*SECDAY*tt0/(SECDAY*pb))*tt0; /* fctn(18+j) */
      else if (param==param_a1dot)
        return (som*(cbe-ecc) + com*sbe*sqrt(tt))*tt0;            /* fctn(24+j) */
      else if (param==param_omdot)
        return asini*(com*(cbe-ecc) - som*sqrt(tt)*sbe)*tt0;      /* fctn(14+j) */
      else if (param==param_edot)                            
        return (-(alpha*(1.0+sbe*sbe-ecc*cbe)*tt - beta*(cbe-ecc)*sbe)*s/tt)*tt0; /* fctn(25+j) */
      else if (param==param_gamma) 
        return sbe;                                               /* fctn(15+j) */
      return 0.0;
    """

def BT_period_gregory(t, P0, P1, PEPOCH, PB, ECC, A1, T0, OM, RA_RAD, DEC_RAD, \
        EDOT=0.0, PBDOT=0.0, OMDOT=0.0):
    """
    The 'BT' binary model for the pulse period. Model:
    Blandford & Teukolsky (1976), ApJ, 205, 580-591

    Adjusted from code by Gregory Desvignes
    NOTE: incompatible with code below

    @param P0:          The pulse period [sec]
    @param P1:          The pulse period derivative [sec/sec]
    @param PEPOCH:      Position EPOCH
    @param PB:          Binary period [days]
    @param ECC:         Eccentricity
    @param A1:          Projected semi-major axis (lt-sec)
    @param T0:          Time of ascending node (TASC)
    @param OM:          Omega (longitude of periastron) [deg]
    @param RA_RAD:      Pulsar position (right ascension) [rad]
    @param DEC_RAD:     Pulsar position (declination) [rad]
    @param EDOT:        Time-derivative of ECC [0.0]
    @param PBDOT:       Time-derivative of PB [0.0]
    @param OMDOT:       Time-derivative of OMEGA [0.0]
    """
    tt0 = (t-T0) * SECS_PER_DAY
    pb = PB * SECS_PER_DAY
    ecc = ECC + EDOT * tt0
    omega = (OM + OMDOT*tt0/(SECS_PER_DAY*365.25)) * DEG2RAD
    pbdot = PBDOT

    if not np.all(np.logical_and(ecc >= 0.0, ecc <= 1.0)):
        return np.inf

    # Get the doppler amplitude
    kappa = 2*np.pi*A1/(PB*SECS_PER_DAY*np.sqrt(1-ecc**2))

    # Obtain the true anomaly through the eccentric anomaly
    orbits = tt0 / pb - 0.5 * pbdot * (tt0 / pb) ** 2
    norbits = np.array(np.floor(orbits), dtype=np.long)
    phase = 2 * np.pi * (orbits - norbits)
    bige = eccentric_anomaly(ecc, phase)
    true_anom = 2*np.arctan(np.sqrt((1+ecc)/(1-ecc))*np.tan(bige/2))

    return 1000*(P0+P1*(t-PEPOCH)*SECS_PER_DAY) * (1+kappa*np.cos(true_anom+omega) )


def BT_period(t, P0, P1, PEPOCH, PB, ECC, A1, T0, OM, RA_RAD, DEC_RAD, \
        EDOT=0.0, PBDOT=0.0, OMDOT=0.0):
    """
    The 'BT' binary model for the pulse period. Model as in:
    W.M. Smart, (1962), "Spherical Astronomy", p359

    See also: Blandford & Teukolsky (1976), ApJ, 205, 580-591

    @param P0:          The pulse period [sec]
    @param P1:          The pulse period derivative [sec/sec]
    @param PEPOCH:      Position EPOCH
    @param PB:          Binary period [days]
    @param ECC:         Eccentricity
    @param A1:          Projected semi-major axis (lt-sec)
    @param T0:          Time of ascending node (TASC)
    @param OM:          Omega (longitude of periastron) [deg]
    @param RA_RAD:      Pulsar position (right ascension) [rad]
    @param DEC_RAD:     Pulsar position (declination) [rad]
    @param EDOT:        Time-derivative of ECC [0.0]
    @param PBDOT:       Time-derivative of PB [0.0]
    @param OMDOT:       Time-derivative of OMEGA [0.0]
    """
    tt0 = (t-T0) * SECS_PER_DAY
    pb = PB * SECS_PER_DAY
    ecc = ECC + EDOT * tt0
    omega = (OM + OMDOT*tt0/(SECS_PER_DAY*365.25)) * DEG2RAD
    pbdot = PBDOT

    if not np.all(np.logical_and(ecc >= 0.0, ecc <= 1.0)):
        return np.inf

    # Calculate the orbital phase
    orbits = tt0 / pb - 0.5 * pbdot * (tt0 / pb) ** 2
    norbits = np.array(np.floor(orbits), dtype=np.long)
    phase = 2 * np.pi * (orbits - norbits)

    # Obtain the true anomaly through the eccentric anomaly
    ea = eccentric_anomaly(ecc, phase)
    ta = 2*np.arctan(np.sqrt((1+ecc)/(1-ecc))*np.tan(ea/2))

    # Projected velocity of the pulsar in the direction of the line-of-sight
    # (Divided by speed of light due to units of A1)
    vl = 2*np.pi*A1/(PB*SECS_PER_DAY*np.sqrt(1-ecc**2))

    # Pulse period, adjusted for frequency evolution
    Px = P0 + P1*(t-PEPOCH)*SECS_PER_DAY

    return Px * (1 + vl * ((np.cos(ta)+ecc)*np.cos(omega)-np.sin(ta)*np.sin(omega)))


def RED_period(t, P0, AMP, OM, T0, ECC, PB):
    """
    A reduced 6-parameter version of the 'BT' binary model for the pulse period.

    @param P0:          The pulse period [sec]
    @param AMP:         Pulse period modulation amplitude [sec]
    @param OM:          Omega (longitude of periastron) [deg]
    @param T0:          Time of ascending node (TASC)
    @param ECC:         Eccentricity
    @param PB:          Binary period [days]
    """
    om = OM * DEG2RAD

    if not np.all(np.logical_and(ECC >= 0.0, ECC <= 1.0)):
        return np.inf

    # Calculate the orbital phase
    orbits = (t-T0) / PB
    norbits = np.array(np.floor(orbits), dtype=np.long)
    phase = 2 * np.pi * (orbits - norbits)

    # Obtain the true anomaly through the eccentric anomaly
    ea = eccentric_anomaly(ECC, phase)
    ta = 2*np.arctan(np.sqrt((1+ECC)/(1-ECC))*np.tan(ea/2))

    return P0 + AMP * np.cos(om+ta)



def DD_delay(t, PB, T0, A1, ECC=0.0, OM=0.0, OMDOT=0.0, am2=0.0, PBDOT=0.0, EDOT=0.0, \
        XDOT=0.0, XPBDOT=0.0, gamma=0.0, kin=None, sini=None):
    """
    Delay due to pulsar binary motion. Model:
    Damour & Deruelle...
    """
    # Sin i
    if kin is not None:
        si = np.sin(kin * DEG2RAD)
    elif sini is not None:
        si = sini
    else:
        si = 0.0

    if si > 1.0:
        print("Sin I > 1.0. Setting to 1: should probably use DDS model")
        si = 1.0

    m2 = am2 * SUNMASS

    pb = PB * SECS_PER_DAY
    an = 2.0*np.pi / pb
    k = OMDOT / (RAD2DEG*365.25*SECS_PER_DAY*an)

    t0 = T0
    ct = t
    tt0 = (ct - t0) * SECS_PER_DAY

    omz = OM
    xdot = XDOT
    pbdot = PBDOT
    edot = EDOT
    xpbdot = XPBDOT

    x = A1 + xdot*tt0
    ecc = ECC + edot*tt0
    er, eth = ecc, ecc

    assert np.all(np.logical_and(ecc >= 0, ecc <= 1)), \
        "BT model: Eccentricity goes out of range"

    orbits = tt0/pb - 0.5*(pbdot+xpbdot)*(tt0/pb)*(tt0/pb)**2
    norbits = np.array(np.floor(orbits), dtype=np.long)
    phase = 2 * np.pi * (orbits - norbits)
    u = eccentric_anomaly(ecc, phase)

    # DD equations: 17b, 17c, 29, and 46 through 52
    su = np.sin(u)
    cu = np.cos(u)
    onemecu = 1.0-ecc*cu
    cae = (cu-ecc)/onemecu
    sae = np.sqrt(1.0-pow(ecc,2))*su/onemecu
    ae = np.arctan2(sae, cae)
    ae[ae<0.0] += 2.0*np.pi
    ae = 2.0 * np.pi * orbits + ae - phase
    omega = omz / RAD2DEG + k * ae
    sw = np.sin(omega)
    cw = np.cos(omega)
    alpha = x * sw
    beta = x * np.sqrt(1-eth**2) * cw
    bg = beta + gamma
    dre = alpha * (cu-er) + bg * su
    drep = -alpha * su + bg * cu
    drepp = -alpha * cu - bg * su
    anhat = an / onemecu

    # DD equations: 26, 27, 57:
    sqr1me2 = np.sqrt(1-ecc**2)
    cume = cu - ecc
    brace = onemecu - si*(sw*cume+sqr1me2*cw*su)
    dlogbr = np.log(brace)
    ds = -2*m2*dlogbr

    #  Now compute d2bar, the orbital time correction in DD equation 42
    d2bar = dre*(1-anhat*drep+(anhat**2)*(drep**2 + 0.5*dre*drepp - \
                      0.5*ecc*su*dre*drep/onemecu)) + ds
    return -d2bar

    """
      if (param==-1) return torb;
      
      /*  Now we need the partial derivatives. Use DD equations 62a - 62k. */
      csigma=x*(-sw*su+sqr1me2*cw*cu)/onemecu;
      ce=su*csigma-x*sw-ecc*x*cw*su/sqr1me2;
      cx=sw*cume+sqr1me2*cw*su;
      comega=x*(cw*cume-sqr1me2*sw*su);
      cgamma=su;
      cdth=-ecc*ecc*x*cw*su/sqr1me2;
      cm2=-2*dlogbr;
      csi=2*m2*(sw*cume+sqr1me2*cw*su)/brace; 
      if (param==param_pb)
        return -csigma*an*SECDAY*tt0/(pb*SECDAY); 
      else if (param==param_a1)
        return cx;
      else if (param==param_ecc)
        return ce;
      else if (param==param_edot)
        return ce*tt0;
      else if (param==param_om)
        return comega;
      else if (param==param_omdot)
        return ae*comega/(an*360.0/(2.0*M_PI)*365.25*SECDAY);
      else if (param==param_t0)
        return -csigma*an*SECDAY;
      else if (param==param_pbdot)
        return 0.5*tt0*(-csigma*an*SECDAY*tt0/(pb*SECDAY));
      else if (param==param_sini)
        return csi;
      else if (param==param_gamma)
        return cgamma;
      else if (param==param_m2)
        return cm2*SUNMASS;
      else if (param==param_a1dot) /* Also known as xdot */
        return cx*tt0;

      return 0.0;
    """

def findCandidates(par, stat, func, args=(), comp='log10', threshold=0.2):
    """
    Given some range of a statistic, find candidate parameter values that
    minimize this statistic.

    @param par:         The parameter values
    @param stat:        Statistic values for values of par
    @param func:        Function to calculate the statistic
    @param args:        Extra arguments to the function func
    @param comp:        Way to compare the statistic
    @param threshold:   Threshold difference for inclusion

    @return:            List of candidate parameter values
    """
    ind = np.argmin(stat)
    if comp=='log10':
        mv = np.log10(stat[ind])
        msk = np.log10(stat) < mv+threshold
    elif comp=='flat':
        mv = stat[ind]
        msk = stat < mv + threshold
    else:
        raise NotImplemented("Unknown comparison string")

    # Select all values below the threshold
    inds = np.nonzero(msk)[0]

    # All neighboring candidates should be counted as one. Group them
    bucket_inds = [[inds[0]]]
    for ii in inds[1:]:
        if ii == bucket_inds[-1][-1]+1:
            bucket_inds[-1].append(ii)
        else:
            bucket_inds.append([ii])

    # For every group of candidates, we should have to minimize the roughness
    candidates = []
    for ii, bucket in enumerate(bucket_inds):
        bucket = np.array(bucket)
        
        # Use the brent minimizer with the following bracket
        if len(bucket) > 1:
            bracket = (par[bucket[0]], par[bucket[-1]])
        else:
            minind = max(0, bucket[0]-1)
            maxind = min(len(par)-1, bucket[0]+1)
            bracket = (par[minind], par[maxind])

        res = so.minimize_scalar(func, \
                bracket=bracket, \
                args=args, method='brent')

        cand = min(max(res.x, bracket[0]), bracket[1])

        candidates.append(cand)

    return np.array(candidates)


class orbitpar(object):
    """
    This class represents parameters for the orbit class
    """
    def __init__(self, name, *args, **kwargs):

        if name == 'START' or name == 'FINISH':
            # Do something else here?
            self.name = name
            self._set = True
            self._fit = False
            self._val = 0.0
            self._err = 0.0
        else:
            self.name = name
            self._set = True
            self._fit = False
            self._val = 0.0
            self._err = 0.0

    @property
    def val(self):
        return self._val

    @val.setter
    def val(self, value):
        self._val = value

    @property
    def err(self):
        return self._err

    @err.setter
    def err(self, value):
        self._err = value

    @property
    def fit(self):
        return self._fit

    @fit.setter
    def fit(self, value):
        self._fit = value

    @property
    def set(self):
        return self._set

    @set.setter
    def set(self, value):
        self.set = value

def createOrbitPar(parname):
    """
    Do we need this function
    """
    newpar = orbitpar(parname)
    return newpar



class orbitpulsar(object):
    """
    Class that can be used to fit for a first-estimate pulsar ephemeris
    """
    def __init__(self, parfilename=None, perfilename=None, ms=False):
        self.initParameters()
        self.readPerFile(perfilename, ms=ms)
        self.readParFile(parfilename)

    def __getitem__(self, key):
        return self.pardict[key]

    def __contains__(self, key):
        return key in self.pardict

    def initParameters(self):
        """
        Initialize the binary parameters
        """
        self.binaryModel = 'BT'
        self.bmparams = OrderedDict()
        self.bmparams['BT'] = ['RA', 'DEC', 'P0', 'P1', 'PEPOCH', 'PB', 'ECC', 'A1', 'T0', 'OM']

        self.parf = None
        self.pardict = OrderedDict()

        for par in self.bmparams[self.binaryModel]:
            newpar = createOrbitPar(par)
            self.pardict[par] = newpar

        self.mjds = np.zeros(0)
        self.periods = np.zeros(0)
        self.periodserrs = np.zeros(0)

    def readPerFile(self, perfilename=None, ms=False):
        """
        Read a period-file, in the PRESO (.bestprof) format

        @param perfilename: PRESO .bestprof file
        """

        if perfilename is not None and os.path.isfile(perfilename):
            self.perfilename = perfilename
            dat = np.loadtxt(perfilename)
            self.mjds = dat[:,0]
            self.periods = dat[:,1]

            if dat.shape[1] > 2:
                # Have uncertainties
                self.periodserrs = dat[:,2]
            else:
                self.periodserrs = None

            if ms:
                self.periods *= 1000.0
        else:
            self.perfilename = None

    def readParFile(self, parfilename):
        """
        Read a par-file

        @param parfilename: timing model parameter file
        """
        if parfilename is not None and os.path.isfile(parfilename):
            self.parf = parfile.Parfile()
            
            self.parf.read(parfilename)

            if isinstance(self.parf.RAJ, basestring):
                self['RA'].val = np.float128(ephem.hours(str(self.parf.RAJ)))
            else:
                self['RA'].val = np.float128(self.parf.RAJ)

            if isinstance(self.parf.DECJ, basestring):
                self['DEC'].val = np.float128(ephem.degrees(str(self.parf.DECJ)))
            else:
                self['DEC'].val = self.parf.DECJ

            self['P0'].val = np.float128(self.parf.P0)
            self['P1'].val = np.float128(self.parf.P1)
            self['PEPOCH'].val = np.float128(self.parf.PEPOCH)
            self['PB'].val = np.float128(self.parf.PB)
            self['ECC'].val = np.float128(self.parf.ECC)
            self['A1'].val = np.float128(self.parf.A1)
            self['T0'].val = np.float128(self.parf.T0)
            self['OM'].val = np.float128(self.parf.OM)
        else:
            self['RA'].val = np.float128(0.0)
            self['DEC'].val = np.float128(0.0)
            self['P0'].val = np.float128(1.0)
            self['P1'].val = np.float128(1.0)
            self['PEPOCH'].val = np.float128(0.0)
            self['PB'].val = np.float128(1.0)
            self['ECC'].val = np.float128(0.0)
            self['A1'].val = np.float128(0.0)
            self['T0'].val = np.float128(0.0)
            self['OM'].val = np.float128(0.0)
            self.parfilename = None

    def writeParFile(self, parfilename):
        """
        Write a par-file

        @param parfilename: timing model parameter file

        @return:    the name of the file that's written.
        """
        filename = parfilename
        while os.path.isfile(filename) or os.path.isdir(filename):
            filename = parfilename + str(np.random.randint(0, 10000))

        if filename is not None and not os.path.isfile(filename) and \
                self.parf is not None:
            self.parf.RAJ = str(ephem.hours(self['RA'].val))
            self.parf.DEC = str(ephem.degrees(self['DEC'].val))
            self.parf.P0 = self['P0'].val
            self.parf.F0 = 1.0/self['P0'].val
            self.parf.P1 = self['P1'].val
            self.parf.F1 = -self['P1'].val/(self['P0'].val**2)
            self.parf.PEPOCH = self['PEPOCH'].val
            self.parf.PB = self['PB'].val
            self.parf.ECC = self['ECC'].val
            self.parf.A1 = self['A1'].val
            self.parf.T0 = self['T0'].val
            self.parf.OM = self['OM'].val

            self.parf.write(filename)
        else:
            raise IOError("Cannot write a valid parfile")

        return filename

    def parmask(self, which='fit', pars=None):
        """
        Return a boolean mask for a given selection of parameters

        @param which:   O
        """
        pars = self.pars(which='set')

        if pars is None:
            pars = self.pars(which='set')

        if which == 'all':
            spars = self.pars(which='all')
        elif which =='set':
            spars = self.pars(which='set')
        elif which == 'fit':
            spars = self.pars(which='fit')

        msk = np.zeros(len(pars), dtype=np.bool)
        for ii, pid in enumerate(pars):
            if pid in spars:
                msk[ii] = True

        return msk

    def vals(self, which='fit', pars=None, newvals=None):
        """
        Given the names of the parameters (None=all), return a numpy array with
        the parameter values

        @param which:   which parameters to obtain/modify
        @param pars:    overrides which, list/tuple of parameters
        @param newvals: numpy array with new values (not used if None)

        @return:    (new) values of selection (numpy array)
        """
        # TODO: Do something with the RA/DEC values here
        msk = self.parmask(which=which, pars=pars)
        rv = np.zeros(np.sum(msk), dtype=np.float128)

        for ii, pd in enumerate(self.pardict):
            if msk[ii]:
                ind = np.sum(msk[:ii+1])-1
                if newvals is not None:
                    self.pardict[pd].val = newvals[ind]
                rv[ind] = np.float128(self.pardict[pd].val)

        return rv

    def pars(self, which='fit'):
        """
        Returns tuple of names of parameters that are fitted/set/etc

        @param which:   Which selection of parameters is requested
                        fit/set/all
        """
        rv = None
        if which == 'fit':
            rv = tuple(key for key in self.pardict if self.pardict[key].fit)
        elif which == 'set':
            rv = tuple(key for key in self.pardict if self.pardict[key].set)
        elif which == 'all':
            rv = tuple(key for key in self.pardict)
        return rv

    def orbitModel(self, mjds=None, pardict=None, parameters=None, which='set',
            parlist=None, model=None):
        """
        Return the model for the pulse period, given the current binary model
        and parameters

        @param mjds:        If not None, use these mjds, instead of the
                            intrinsic ones
        @param pardict:     If not None, use these parameters, instead of the
                            intrinsic ones
        @param parameters:  Overrides pardict. If not None, use this array of
                            parameters instead of the intrinsic ones
        @param which:       If parameters is set, this indicator set which
                            parameters are actually in the array
        @param parlist:     (Overrides which) which parameters are in the array
        @param model:       Which binary model to use (None = self.binaryModel)
        """
        if mjds is None:
            mj = self.mjds
        else:
            mj = mjds

        if parameters is not None:
            mask = self.parmask(which=which, pars=parlist)
            apars = self.vals(which='set')
            apars[mask] = parameters
            pardict = array_to_pardict(apars, which=self.binaryModel)
        elif pardict is not None:
            pass
        else:
            pardict = self.pardict

        if model is None:
            model = self.binaryModel

        bmarr = pardict_to_array(pardict, which=model)
        pmodel = np.zeros(len(self.mjds))

        if model == 'BT':
            pmodel = BT_period(mj, *bmarr)
        elif model == 'BTG':
            pmodel = BT_period_gregory(mj, *bmarr)
        elif model == 'RED':
            pmodel = RED_period(mj, *bmarr)
        elif model  == 'DD':
            raise NotImplemented("Only BT works for now")

        return pmodel

    def orbitResiduals(self, pardict=None, parameters=None, which='set', \
            parlist=None, weight=False, model=None):
        """
        Return the residuals = data - model for the pulse period, given the
        current binary model and parameters

        @param pardict:     If not None, use these parameters, instead of the
                            intrinsic ones
        @param parameters:  Overrides pardict. If not None, use this array of
                            parameters instead of the intrinsic ones
        @param which:       If parameters is set, this indicator set which
                            parameters are actually in the array
        @param parlist:     (Overrides which) which parameters are in the array
        @param weight:      If True, weight the residuals by their uncertainties
        @param model:       Which period model to use (Default self.binaryModel)
        """
        if parameters is not None:
            mask = self.parmask(which=which, pars=parlist)
            apars = self.vals(which='set')
            apars[mask] = parameters
            pardict = array_to_pardict(apars, which=self.binaryModel)
        elif pardict is not None:
            pass
        else:
            pardict = self.pardict

        resids = self.periods - self.orbitModel(pardict=pardict, model=model)

        if weight and self.periodserrs is not None:
            resids /= self.periodserrs

        return resids

    def loglikelihood(self, pardict=None, parameters=None, which='set',
            parlist=None, model=None):
        """
        Return the log-likelihood for the data, given the model

        @param pardict:     If not None, use these parameters, instead of the
                            intrinsic ones
        @param parameters:  Overrides pardict. If not None, use this array of
                            parameters instead of the intrinsic ones
        @param which:       If parameters is set, this indicator set which
                            parameters are actually in the array
        @param parlist:     (Overrides which) which parameters are in the array
        @param model:       Which period model to use (Default self.binaryModel)
        @return:    The log-likelihood value
        """
        if self.periodserrs is None:
            raise ValueError("Likelihood requires uncertainties")

        n = len(self.periods)
        xi2 = np.sum(self.orbitResiduals(pardict=pardict, \
                parameters=parameters, which=which, parlist=parlist, \
                weight=True, model=model)**2)
        return -0.5*xi2 - np.sum(np.log(self.periodserrs)) - 0.5*n*np.log(2*np.pi)

    def simData(self, mjds, perr=1.2e-7, model=None):
        """
        For the current timing model, generate mock observed periods

        @param mjds:    Array with new observation times
        @param perr:    Uncertainty of the measurements
        @param model:       Which period model to use (Default self.binaryModel)
        """
        nobs = len(mjds)
        self.periodserrs = np.ones(nobs) * perr
        self.mjds = np.array(mjds)
        self.periods = self.orbitModel(mjds = self.mjds, model=model)

    def roughness_fast(self, pb):
        """
        Calculate the roughness, given an array of binary periods (vectorized
        version)

        Using the Roughness as defined in:
        Bhattacharyya & Nityanada, 2008, MNRAS, 387, Issue 1, pp. 273-278

        @param pb:  Array with binary periods to try
        """
        n = len(pb)
        mjds, per = np.meshgrid(self.mjds, pb)
        phi = np.fmod(np.float64(self['T0'].val), per)
        mjds = mjds-phi

        periods = self.periods.reshape(len(self.periods), 1).repeat(n, axis=1).T
        phase = np.fmod(mjds, per) / per

        ps = np.take(periods, np.argsort(phase, axis=1))
        R = np.sum((ps[:,:-1]-ps[:,1:])**2, axis=1)

        return R

    def roughness_slow(self, pb):
        """
        Calculate the roughness, given an array of binary periods

        Using the Roughness as defined in:
        Bhattacharyya & Nityanada, 2008, MNRAS, 387, Issue 1, pp. 273-278
        """
        n = len(pb)
        R = np.zeros(n)

        for ii, per in enumerate(pb):
            phi = np.fmod(np.float64(self['T0'].val), per)
            phase = np.fmod(self.mjds-phi, per) / per
            inds = np.argsort(phase)

            R[ii] = np.sum((self.periods[inds][:-1] - self.periods[inds][1:])**2)

        return R

    def roughness_new(self, pb, scale=0.3):
        """
        Calculate the roughness, given an array of binary periods

        Using a modified version of the Roughness as defined in:
        Bhattacharyya & Nityanada, 2008, MNRAS, 387, Issue 1, pp. 273-278

        @param pb:      Array with binary periods to try
        @param scale:   Scale up to which we are sensitive to changes

        TODO: Test more, the scale value is not obvious.
        """
        n = len(pb)
        mjds, per = np.meshgrid(self.mjds, pb)
        phi = np.fmod(np.float64(self['T0'].val), per)
        mjds = mjds-phi

        periods = self.periods.reshape(len(self.periods), 1).repeat(n, axis=1).T
        phase = np.fmod(mjds, per) / per

        if self.periodserrs is not None:
            errs = self.periodserrs.reshape(len(self.periodserrs), 1).repeat(n,
                    axis=1).T
            inds = np.argsort(phase, axis=1)
            ps = np.take(periods, inds)
            es = np.take(errs, inds)
            hs = np.take(phase, inds)

            R = np.sum( ((ps[:,:-1]-ps[:,1:])**2/(es[:,:-1]*es[:,1:])) /\
                    (scale**2+(hs[:,:-1]-hs[:,1:])**2), axis=1)
        else:
            ps = np.take(periods, np.argsort(phase, axis=1))
            R = np.sum((ps[:,:-1]-ps[:,1:])**2, axis=1)

        return R


    def roughness(self, pb):
        """
        Calculate the roughness for binary period pb (array)
        """
        return self.roughness_new(pb)

    def roughnessPlot(self, pbmin=0.007, pbmax=None, frac=0.01):
        """
        Calculate the roughness plot

        @param pbmin:   Minimum orbital period to search (7e-3 days = 10min)
        @param pbmax:   Maximum orbital period to search (Tmax)
        @param frac:    Minimum fractional change to scan (0.02)

        @return: pb, roughness
        """
        if pbmax is None:
            Tmax = np.max(self.mjds) - np.min(self.mjds)
            pbmax = Tmax

        Ntrials = int( ((pbmax-pbmin)*Tmax*2*np.pi/frac)**(1.0/3.0)/pbmin )
        ind = np.arange(Ntrials)
        dpb = frac * pbmin**3/(2*np.pi*Tmax)
        pb = pbmin + dpb * ind**3
        rg = self.roughness(pb)

        return pb, rg

    def PbEst(self, pbmin=0.007, pbmax=None, frac=0.01):
        """
        Return an estimate of the binary period using the roughness

        @return Pb
        """
        pb, rg = self.roughnessPlot(pbmin, pbmax, frac)

        return pb[np.argmin(rg)]


    def PbCandidates(self, pbmin=0.007, pbmax=None, frac=0.01, threshold=0.2):
        """
        Return the candidate binary periods, based on the roughness.

        @param pbmin:       Minimum binary period (default 0.007 s^{-1})
        @param pbmax:       Maximum binary period (default None -> 1/Tmax)
        @param frac:        1/Sampling density of the binary period (lower = more)
        @param threshold:   If log10(rg) for is within this distance from
                            log10(rg_min), this roughness is a candidate
        
        @return:        List of candidate periods
        """
        pb, rg = self.roughnessPlot(pbmin, pbmax, frac)

        def funcPb(pb, psr):
            return psr.roughness(np.array([pb]))[0]
                
        return np.array(findCandidates(pb, rg, funcPb, args=(self,), \
                comp='log10', threshold=threshold))

    def Peo(self, Pb=None, T0=None, kind='linear'):
        """
        Return the Peven and Podd functions, as defined in:
        Bhattacharyya & Nityanada, 2008, MNRAS, 387, Issue 1, pp. 273-278
        For the interpolation, use a method indicated by kind
        
        @param Pb:  Estimate of the binary period. If not set, use an estimate
                    obtained through the roughness
        @param T0:  Estimate of periastron passage. If not set, use the current
                    value
        #param:     What kind of interpolation to use ('cubic', 'linear', etc.)

        @return:    Peven, Podd
        """
        if len(self.periods) < 10:
            raise ValueError("Need more than 10 observations")

        if Pb is None:
            Pb = self.PbEst()

        if T0 is None:
            T0 = np.float64(self['T0'].val)

        phi = np.fmod(T0, Pb)
        phase = np.fmod(self.mjds-phi, Pb) / Pb
        ind = np.argsort(phase)

        phase = phase[ind]
        periods = self.periods[ind]

        # In order to get correct estimates, we wrap around the phase with three
        # extra points each way
        phase_w = np.append(np.append(phase[-3:]-1.0, phase), phase[:3]+1.0)
        periods_w = np.append(np.append(periods[-3:], periods), periods[:3])

        func = si.interp1d(phase_w, periods_w, kind=kind)

        Peven = 0.5 * (func(phase) + func(1.0-phase))
        Podd = 0.5 * (func(phase) - func(1.0-phase))

        return Peven, Podd

    def Peo_interp(self, Pb=None, T0=None, kind='linear'):
        """
        Return the interpolated Peven and Podd for the full phase coverage
        (for debugging purposes)
        
        @param Pb:  Estimate of the binary period. If not set, use an estimate
                    obtained through the roughness
        @param T0:  Estimate of periastron passage. If not set, use the current
                    value
        #param:     What kind of interpolation to use ('cubic', 'linear', etc.)

        @return:    Peven, Podd
        """
        if len(self.periods) < 10:
            raise ValueError("Need more than 10 observations")

        if Pb is None:
            Pb = self.PbEst()

        if T0 is None:
            T0 = np.float64(self['T0'].val)

        phi = np.fmod(T0, Pb)
        phase = np.fmod(self.mjds-phi, Pb) / Pb
        ind = np.argsort(phase)

        phase = phase[ind]
        periods = self.periods[ind]

        # In order to get correct estimates, we wrap around the phase with three
        # extra points each way
        phase_w = np.append(np.append(phase[-3:]-1.0, phase), phase[:3]+1.0)
        periods_w = np.append(np.append(periods[-3:], periods), periods[:3])

        func = si.interp1d(phase_w, periods_w, kind=kind)

        newphase = np.linspace(0.01, 0.99, 200)
        Peven = 0.5 * (func(newphase) + func(1.0-newphase))
        Podd = 0.5 * (func(newphase) - func(1.0-newphase))

        return Peven, Podd, newphase, func(newphase)

    def oe_ABC_leastsq(self, Peven, Podd):
        """
        Calculate OM, P0, and amplitude from Peven and Podd, using a
        least-squares fit to the hodograph. Use the method described in:
        Bhattacharyya & Nityanada, 2008, MNRAS, 387, Issue 1, pp. 273-278

        @param Peven:   Peven
        @param Podd:    Podd

        @return OM, RAMP, P0, xi2
        """
        if len(Peven) != len(Podd) or len(Peven) < 1:
            raise ValueError("Peven and Podd not compatible with fit")
        n = len(Peven)
        Pmean = np.mean(Peven)

        # Create the design matrix (n x 3)
        # We'll solve the equation A*Podd**2 + B*Peven**2 + C*Peven = 1
        M = np.array([Podd**2, (Peven-Pmean)**2, Peven-Pmean]).T
        y = np.ones(n)

        # Perform the least-squares fit using an SVD
        MMt = np.dot(M.T, M)
        cf = sl.cho_factor(MMt)
        Mi = sl.cho_solve(cf, M.T)
        x = np.dot(Mi, y)                   # x = [A, B, C]
        xi2 = np.sum((np.dot(M, x) - y)**2)
    
        # Force both A and B to be positive
        if np.all(x[:2] > 0.0):
            xi2 = np.sum((np.dot(M, x) - y)**2)
            OM = np.fmod(360.0+np.arctan(np.sqrt(x[1]/x[0])) * RAD2DEG, 360.0)
            P0 = -0.5*x[2]/x[1] + Pmean
            RAMP = np.sqrt(1.0/x[0]+1.0/x[1]+0.25*x[2]**2*(1.0/(x[0]*x[1])+1.0/x[1]**2))
        else:
            OM, P0, RAMP, xi2 = 0.0, 0.0, 0.0, np.inf

        return OM, P0, RAMP, xi2

    def BNscan(self, Pb=None, T0=None):
        """
        Use the Bhattacharyya & Nityanada method to estimate: Pb, T0, OM, RAMP.
        Adapted from:
        Bhattacharyya & Nityanada, 2008, MNRAS, 387, Issue 1, pp. 273-278

        @param Pb:  Estimated value for binary period
        @param T0:  Trial values of T0 (set here if None)

        @return: Pb (scalar), T0, OM, RAMP, P0, xi2
        """
        if Pb is None:
            Pb = self.PbEst()

        if T0 is None:
            N = 1000
            T0 = np.linspace(\
                    np.float64(self['T0'].val)-0.25*Pb, \
                    np.float64(self['T0'].val)+0.25*Pb, N)
        else:
            N = len(T0)

        xi2 = np.zeros(N)
        OM = np.zeros(N)
        RAMP = np.zeros(N)
        P0 = np.zeros(N)
        for ii, t in enumerate(T0):
            Peven, Podd = self.Peo(Pb=Pb, T0=t)
            OM[ii], P0[ii], RAMP[ii], xi2[ii] = self.oe_ABC_leastsq(Peven, Podd)

        return Pb, T0, OM, P0, RAMP, xi2

    def BNest(self, T0=None):
        """
        Use the Bhattacharyya & Nityanada method to estimate: Pb, T0, OM, P0.
        amplitude
        Adapted from:
        Bhattacharyya & Nityanada, 2008, MNRAS, 387, Issue 1, pp. 273-278

        @param T0:  Trial values of T0 (set here if None)

        @return: Pb, T0, OM, P0, RAMP
        """

        def BNfunc(T0, Pb, psr):
            """
            Calculate the xi2 for the Peven/Podd fit for specific T0/Pb combo
            """
            Peven, Podd = psr.Peo(Pb=Pb, T0=T0)
            OM, P0, RAMP, xi2 = psr.oe_ABC_leastsq(Peven, Podd)
            return xi2


        Pb = self.PbEst()

        if T0 is None:
            N = 1000
            T0 = np.linspace(\
                    np.float64(self['T0'].val)-0.25*Pb, \
                    np.float64(self['T0'].val)+0.25*Pb, N)
        else:
            N = len(T0)

        Pb, T0, OM, P0, RAMP, xi2 = self.BNscan(Pb, T0)

        ind = np.argmin(xi2)

        minind = max(0, ind-1)
        maxind = min(len(xi2), ind+1)

        # Do a SciPy optimization procedure
        bracket = (T0[minind], T0[maxind])
        res = so.minimize_scalar(BNfunc, \
                bracket=bracket, \
                args=(Pb, self), method='brent')

        T0 = res.x
        Peven, Podd = self.Peo(Pb=Pb, T0=T0)
        OM, P0, RAMP, xi2 = self.oe_ABC_leastsq(Peven, Podd)

        return Pb, T0, OM, P0, RAMP

    def BNcandidates(self, PbC=None, T0C=None, threshold=0.25):
        """
        Estimate candidates for the parameters: Pb, T0, OM, P0, amplitude (A1)
        Adapted from:
        Bhattacharyya & Nityanada, 2008, MNRAS, 387, Issue 1, pp. 273-278

        @param PbC:         Orbital period candidates (set here if None)
        @param T0:          Trial values of T0 (set here if None)
        @param threshold:   Log10-threshold for candidate inclusion

        @return: 2D array, with rows [P0, RAMP, OM, T0, ECC, PB]
        """
        if PbC is None:
            PbC = self.PbCandidates(threshold=threshold)
        
        # Scan/search over these time of ascending node passages
        if T0C is None:
            N = 250
            T0C = np.linspace(\
                    np.float64(self['T0'].val)-0.25*PbC[0], \
                    np.float64(self['T0'].val)+0.25*PbC[0], N)
        else:
            N = len(T0C)
        
        # Lists of all the parameter candidates
        Pb_l, T0_l, OM_l, P0_l, RAMP_l, ECC_l = [], [], [], [], [], []

        # Ellipse-fit xi^2 function to minimize wrt T0
        def BNfunc(T0, Pb, psr):
            """
            Calculate the xi2 for the Peven/Podd fit for specific T0/Pb combo
            """
            Peven, Podd = psr.Peo(Pb=Pb, T0=T0)
            OM, P0, RAMP, xi2 = psr.oe_ABC_leastsq(Peven, Podd)
            return xi2
        
        def nloglik(ecc, P0, RAMP, OM, T0, PB, psr):
            pars = np.array([P0, RAMP, OM, T0, ecc, PB])
            pd = array_to_pardict(pars, which='RED')
            return -psr.loglikelihood(pardict=pd, model='RED')
        
        # For every orbital period candidate, find T0 candidates
        for ii, Pbe in enumerate(PbC):
            # Scan over the T0 values
            PbX, T0s, OMs, P0s, RAMPs, xi2 = self.BNscan(Pbe, T0C)
            
            # Find local minima, and produce candidates of T0
            T0CC = findCandidates(T0C, xi2, BNfunc, args=(Pbe, self), comp='log10', threshold=0.1)

            # For every PB/T0 candidate, obtain the parameter fits, and add to the lists
            for T0_cand in T0CC:
                Peven, Podd = self.Peo(Pb=Pbe, T0=T0_cand)
                OMc, P0, RAMP, xi2 = self.oe_ABC_leastsq(Peven, Podd)

                if RAMP < 0.0:
                    RAMP = -RAMP
                    OMc += 180.0

                # Convert/mirror OM to all quadrants
                OMs = [np.fmod(720.0+OMc, 360.0), np.fmod(900-OMc, 360.0), np.fmod(720-OMc, 360.0), np.fmod(900+OMc, 360.0)]
                
                for OM in OMs:
                    # Now find the eccentricity for each candidate
                    eccp = np.linspace(0.0, 1.0, 100, endpoint=False)
                    args = (P0, RAMP, OM, T0_cand, Pbe, self)
                    nll = np.array([nloglik(ecc, *args) for ecc in eccp])

                    # Make a list with eccentricity candidates
                    ecc_cand = findCandidates(eccp, nll, nloglik, args=args, comp='flat', threshold=0.5)

                    # Add all the candidates to the lists
                    for ecc in ecc_cand:
                        Pb_l.append(Pbe)
                        T0_l.append(T0_cand)
                        OM_l.append(OM)
                        P0_l.append(P0)
                        RAMP_l.append(RAMP)
                        ECC_l.append(ecc)
                
        return np.array([P0_l, RAMP_l, OM_l, T0_l, ECC_l, Pb_l]).T

    def haveCandidate(self, cand, sols, pcovs):
        """
        Check whether the candidate solution is already present in the
        collection of solutions 'sols' with covariances 'pcovs'.

        @param cand:    Candidate solution
        @param sols:    List of already included solutions
        @param pcovs:   List of parameter covariances

        @return:        False if candidate is new, True if already present
        """
        haveC = False

        for ii, sol in enumerate(sols):
            pcov = pcovs[ii]
            if pcov is not None:
                cf = sl.cho_factor(pcov)
                xi2 = np.dot(cand-sol, sl.cho_solve(cf, cand-sol))
                if xi2 <= 9.236:
                    # At 90% confidence with 5-dof xi2 test, same solution
                    haveC = True

        return haveC


    def reduceCandidates(self, cands, ll_threshold=3.0):
        """
        Given a list of parameter candidates, as produced with BNcandidates,
        optimize and reduce the candidates to unique proposals.

        @param cands:           Parameter candidates
        @param ll_threshold:    loglik > max_ll - (10**ll_threshold) * len(toas)
        
        @return:        New parameter candidates
        """
        ll = []
        for cand in cands:
            pd = array_to_pardict(cand, which='RED')
            ll.append(self.loglikelihood(pardict=pd, model='RED'))

        # Which ll-candidates to consider (3.0 is a pretty broad inclusion rate)
        ll = np.array(ll) - np.max(ll)
        msk = (ll > - (10**ll_threshold) * len(self.mjds))

        # Residuals (for leastsq function)
        def resids(pars, psr):
            pd = array_to_pardict(pars, which='RED')
            return psr.orbitResiduals(pardict=pd, weight=True, model='RED')

        # For all selected candidates, perform an optimization
        sols = []
        pcovs = []
        for cand in cands[msk, :]:
            plsq = so.leastsq(resids, np.float64(cand), args=(self), full_output=True)

            if not self.haveCandidate(plsq[0], sols, pcovs):
                sols.append(plsq[0])
                pcovs.append(plsq[1])

        return np.array(sols)



