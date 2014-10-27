#!/usr/bin/env python

import numpy as Num
try:
    from pyslalib import *
except :
#except ImportError:
    print "Cannot import slalib"


def deltav(mjd, ra , dec, dra , ddec, equinoxe):

    # Spherical coordinates to Cartesian coordinates position 'pos'
    pos1 = sla_dcs2c(ra, dec)  
    pos2 = sla_dcs2c(ra+dra, dec+ddec) 

    # Barycentric and heliocentric velocity and position of the Earth
    (pos_helio, velo_helio, pos_bary, velo_bary) = sla_epv(mjd) # Return PH, VH, PB, VB (V in AU s-1)
    (velo_bary, pos_bary, velo_helio, pos_helio) = sla_evp(mjd, equinoxe) # Return fast VB, PB, VH, PH (V in AU d-1)
  
    # Difference of scalar products of two 3-vectors
    return 149600e6*(sla_dvdv(velo_bary, pos2) - sla_dvdv(velo_bary, pos1)) 
    #return (sla_dvdv(velo_bary*86400.0, pos2) - sla_dvdv(velo_bary*86400.0, pos1)) 

def eccentric_anomaly(E, mean_anomaly):
    """
    eccentric_anomaly(mean_anomaly):
            Return the eccentric anomaly in radians, given a set of mean_anomalies
            in radians.
    """
    ma = Num.fmod(mean_anomaly, 2*Num.pi)
    ma = Num.where(ma < 0.0, ma+2*Num.pi, ma)
    eccentricity = E
    ecc_anom_old = ma
    #print ma
    ecc_anom = ma + eccentricity*Num.sin(ecc_anom_old)
    iter = 0
    # This is a simple iteration to solve Kepler's Equation

    if (Num.alen(ecc_anom) >1):
      while (Num.maximum.reduce(Num.fabs(ecc_anom-ecc_anom_old)) > 5e-15):
        ecc_anom_old = ecc_anom
        ecc_anom = ma + eccentricity*Num.sin(ecc_anom_old)
        #print ecc_anom, iter
        iter+=1

    elif(Num.alen(ecc_anom) ==1):
      while (Num.fabs(ecc_anom-ecc_anom_old) > 5e-15):
        ecc_anom_old = ecc_anom
        ecc_anom = ma + eccentricity*Num.sin(ecc_anom_old)
        #print ecc_anom, iter
        iter+=1
    """

    while (Num.fabs(ecc_anom-ecc_anom_old) > 5e-15):
        ecc_anom_old = ecc_anom
        ecc_anom = ma + eccentricity*Num.sin(ecc_anom_old)
    """        

    return ecc_anom

