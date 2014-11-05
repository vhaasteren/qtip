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

from utils import eccentric_anomaly
import fitorbit_parfile as parfile

try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

# Use long doubles for the calculations (from fitorbit)
DEG2RAD    = np.float128('1.7453292519943295769236907684886127134428718885417e-2')
RAD2DEG    = np.float128('57.295779513082320876798154814105170332405472466564')
C           = np.float128('2.99792458e8')


def pardict_to_array(pardict, which='BT'):
    """
    Given a dictionary of parameter values, return an array with the parameters
    in the right order (so it can be used for the binary models)

    @param pardict:     Dictionary of orbit model parameters
    @param which:       Which binary model we are arranging for

    @return x:          Array of model parameters
    """
    x = None
    if which=='BT':
        x = np.zeros(12, dtype=np.float128)
        if 'RA' in pardict:
            if isinstance(pardict['RA'].val, basestring):
                x[0] = np.float128(ephem.hours(str(pardict['RA'].val)))
            else:
                x[0] = np.float128(pardict['RA'].val)

        if 'DEC' in pardict:
            if isinstance(pardict['DEC'].val, basestring):
                x[1] = np.float128(ephem.degrees(str(pardict['DEC'].val)))
            else:
                x[1] = np.float128(pardict['DEC'].val)

        if 'P0' in pardict:
            x[2] = np.float128(pardict['P0'].val)

        if 'P1' in pardict:
            x[3] = np.float128(pardict['P1'].val)

        if 'PEPOCH' in pardict:
            x[4] = np.float128(pardict['PEPOCH'].val)

        if 'PB' in pardict:
            x[5] = np.float128(pardict['PB'].val)

        if 'ECC' in pardict:
            x[6] = np.float128(pardict['ECC'].val)

        if 'A1' in pardict:
            x[7] = np.float128(pardict['A1'].val)

        if 'T0' in pardict:
            x[8] = np.float128(pardict['T0'].val)

        if 'OM' in pardict:
            x[9] = np.float128(pardict['OM'].val)

        if 'RA' in pardict:
            if isinstance(pardict['RA'].val, basestring):
                x[10] = np.float128(ephem.hours(str(pardict['RA'].val)))
            else:
                x[10] = np.float128(pardict['RA'].val)

        if 'DEC' in pardict:
            if isinstance(pardict['DEC'].val, basestring):
                x[11] = np.float128(ephem.degrees(str(pardict['DEC'].val)))
            else:
                x[11] = np.float128(pardict['DEC'].val)
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
    if which=='BT':
        #def BT_period(t, DRA_RAD, DDEC_RAD, P0, P1, PEPOCH, PB, ECC, A1, T0, \
        #        OM, RA_RAD, DEC_RAD):
        pardict['RA'] = createOrbitPar('RA')
        pardict['RA'].val = parameters[0]

        pardict['DEC'] = createOrbitPar('DEC')
        pardict['DEC'].val = parameters[1]

        pardict['P0'] = createOrbitPar('P0')
        pardict['P0'].val = parameters[2]

        pardict['P1'] = createOrbitPar('P1')
        pardict['P1'].val = parameters[3]

        pardict['PEPOCH'] = createOrbitPar('PEPOCH')
        pardict['PEPOCH'].val = parameters[4]

        pardict['PB'] = createOrbitPar('PB')
        pardict['PB'].val = parameters[5]

        pardict['ECC'] = createOrbitPar('ECC')
        pardict['ECC'].val = parameters[6]

        pardict['A1'] = createOrbitPar('A1')
        pardict['A1'].val = parameters[7]

        pardict['T0'] = createOrbitPar('T0')
        pardict['T0'].val = parameters[8]

        pardict['OM'] = createOrbitPar('OM')
        pardict['OM'].val = parameters[9]

        #pardict['XX'] = createOrbitPar('XX')
        #pardict['XX'].val = parameters[10]

        #pardict['XX'] = createOrbitPar('XX')
        #pardict['XX'].val = parameters[11]
    elif which=='DD':
        pass

    return pardict

def BT_period(t, DRA_RAD, DDEC_RAD, P0, P1, PEPOCH, PB, ECC, A1, T0, \
        OM, RA_RAD, DEC_RAD):
    """
    The 'BT' binary model for the pulse period

    @param DRA_RAD:     ??
    @param DDEC_RAD:    ??
    @param P0:          The pulse period [sec]
    @param P1:          The pulse period derivative [sec/sec]
    @param PEPOCH:      Position EPOCH
    @param PB:          Binary period [days]
    @param ECC:         Eccentricity
    @param A1:          ??
    @param T0:          Time of ascending node (TASC)
    @param OM:          Omega (longitude of periastron) [deg]
    @param RA_RAD:      Pulsar position (right ascension) [rad]
    @param DEC_RAD:     Pulsar position (declination) [rad]
    """
    # TODO: Properly define all the variables, and include equations of all
    #       quantities
    if ECC < 0.0:
        return np.inf

    k1 = 2*np.pi*A1/(PB*86400.0*np.sqrt(1-ECC*ECC))

    # Calc easc in rad
    easc = 2*np.arctan(np.sqrt((1-ECC)/(1+ECC)) * np.tan(-OM*DEG2RAD/2))
    #print easc
    epperias = T0 - PB/360.0*(RAD2DEG * easc - RAD2DEG * ECC * np.sin(easc))
    #print t,epperias
    mean_anom = 360*(t-epperias)/PB
    mean_anom = np.fmod(mean_anom,360.0)
    #if mean_anom<360.0:
    #  mean_anom+=360.0
    mean_anom = np.where(np.greater(mean_anom, 360.0), mean_anom-360.0, mean_anom)
        
    # Return ecc_anom (in rad) by iteration
    ecc_anom = eccentric_anomaly(ECC, mean_anom*DEG2RAD)

    # Return true anomaly in deg
    true_anom = 2*RAD2DEG*np.arctan(np.sqrt((1+ECC)/(1-ECC))*np.tan(ecc_anom/2))

    #print "easc=%f  epperias=%f  mean_anom=%f  ecc_anom=%f  true_anom=%f"%(easc,epperias,mean_anom,ecc_anom,true_anom)
    #sys.exit()

    #print RA, DEC
    #dv = deltav(t, RA, DEC, RA-DRA, DEC-DDEC, 2000.0)
    #print dv

    return 1000*(P0+P1*1e-15*(t-PEPOCH)*86400) * (1+k1*np.cos(DEG2RAD*(true_anom+OM)) )
    #return 1000*(P0+P1*1e-15*(t-PEPOCH)*86400) * (1+k1*np.cos(DEG2RAD*(true_anom+OM) + k1*ECC*np.cos(OM)) ) * (1-dv/3e8)
    #return 1000*(P0+P1*1e-15*(t-PEPOCH)*86400) * (1+k1*np.cos(DEG2RAD*(true_anom+OM)) ) * (1-20000/C)


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
        # TODO: Check the units here
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
            self['P1'].val = np.float128(self.parf.P1/1e-15)
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

    def parmask(self, which='fit', pars=None):
        """
        Return a boolean mask for a given selection of parameters
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
            parlist=None):
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

        bmarr = pardict_to_array(pardict, which=self.binaryModel)
        pmodel = np.zeros(len(self.mjds))

        if self.binaryModel == 'BT':
            pmodel = BT_period(mj, *bmarr)
        elif self.binaryModel == 'DD':
            raise NotImplemented("Only BT works for now")

        return pmodel

    def orbitResiduals(self, pardict=None, parameters=None, which='set', \
            parlist=None, weight=False):
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

        resids = self.periods - self.orbitModel(pardict=pardict)

        if weight and self.periodserrs is not None:
            resids /= self.periodserrs

        return resids

    def loglikelihood(self, pardict=None, parameters=None, which='set',
            parlist=None):
        """
        Return the log-likelihood for the data, given the model

        @param pardict:     If not None, use these parameters, instead of the
                            intrinsic ones
        @param parameters:  Overrides pardict. If not None, use this array of
                            parameters instead of the intrinsic ones
        @param which:       If parameters is set, this indicator set which
                            parameters are actually in the array
        @param parlist:     (Overrides which) which parameters are in the array
        @return:    The log-likelihood value
        """
        if self.periodserrs is None:
            raise ValueError("Likelihood requires uncertainties")

        n = len(self.periods)
        xi2 = np.sum(self.orbitResiduals(pardict=pardict, \
                parameters=parameters, which=which, parlist=parlist, \
                weight=True)**2)
        return -0.5*xi2 - np.sum(np.log(self.periodserrs)) - 0.5*n*np.log(2*np.pi)


    def roughness_old(self, pb):
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

    def roughness_new(self, pb):
        """
        Calculate the roughness, given an array of binary periods

        Using a modified version of the Roughness as defined in:
        Bhattacharyya & Nityanada, 2008, MNRAS, 387, Issue 1, pp. 273-278
        """
        n = len(pb)
        R = np.zeros(n)

        for ii, per in enumerate(pb):
            phi = np.fmod(self['T0'].val, per)
            phase = np.fmod(self.mjds-phi, per)
            inds = np.argsort(phase)

            phase = phase[inds]
            periods = self.periods[inds]
            periodserrs = self.periodserrs[inds]
            #R[ii] = np.sum((self.periods[inds][:-1] - self.periods[inds][1:])**2)
            R[ii] = per**2*np.sum((
                        periodserrs[:-1]*periodserrs[1:] +
                        (periods[:-1] - periods[1:])**2) /
                        ((phase[:-1]-phase[1:])**2))

        return R

    def roughness(self, pb):
        """
        Calculate the roughness for binary period pb
        """
        return self.roughness_old(pb)

    def PbEst(self):
        """
        Return an estimate of the binary period using the roughness

        @return Pb
        """
        Ntrials = 25000
        pb = 10**(np.linspace(np.log10(1.0e-4), np.log10(1.0e4), Ntrials))
        rg = self.roughness(pb)
        
        minind = np.argmin(rg)
        return pb[minind]

    def Peo(self, Pb=None, T0=None):
        """
        Return the Peven and Podd functions, as defined in:
        Bhattacharyya & Nityanada, 2008, MNRAS, 387, Issue 1, pp. 273-278
        For the interpolation, use a cubic spline
        
        @param Pb:  Estimate of the binary period. If not set, use an estimate
                    obtained through the roughness
        @param T0:  Estimate of periastron passage. If not set, use the current
                    value

        @return:    Peven, Podd
        """
        if len(self.periods) < 10:
            raise ValueError("Need more than 10 observations")

        if Pb is None:
            Pb = self.PbEst()

        if T0 is None:
            T0 = self['T0'].val

        phi = np.fmod(T0, Pb)
        phase = np.fmod(self.mjds-phi, Pb) / Pb

        # In order to get correct estimates, we wrap around the phase with three
        # extra points each way
        phase_w = np.append(np.append(phase[-3:]-1.0, phase), phase[:3]+1.0)
        periods_w = np.append(np.append(self.periods[-3:], self.periods), \
                self.periods[:3])

        func = si.interp1d(phase_w, periods_w, kind='cubic')

        Peven = 0.5 * (func(phase) + func(1.0-phase))
        Podd = 0.5 * (func(phase) - func(1.0-phase))

        return Peven, Podd

    def oe_ABC_leastsq(self, Peven, Podd):
        """
        Calculate OM and ECC from Peven and Podd, using a least-squares fit to
        the hodograph. Use the method described in:
        Bhattacharyya & Nityanada, 2008, MNRAS, 387, Issue 1, pp. 273-278

        @param Peven:   Peven
        @param Podd:   Podd

        @return OM, ECC, xi2
        """
        if len(Peven) != len(Podd) or len(Peven) < 1:
            raise ValueError("Peven and Podd not compatible with fit")
        n = len(Peven)

        # Create the design matrix (n x 3)
        M = np.array([Podd**2, Peven**2, Peven]).T
        y = np.ones(n)

        # Perform the least-squares fit using an SVD
        MMt = np.dot(M.T, M)
        U, s, Vt = sl.svd(MMt)
        Mi = np.dot(Vt.T * 1.0/s, np.dot(U.T, M.T))
        x = np.dot(Mi, y)

        # x = [A, B, C]
        # Below equation A4
        print(x[0], x[1])
        OM = np.arctan(np.sqrt(x[1]/x[0])) * RAD2DEG
        #ECC = x[2] / np.sqrt(2*x[1]*(2-x[1]))
        ECC = -x[2] * np.sqrt(2*x[1]/(2-x[2])) / (2*x[1])

        xi2 = np.sum((np.dot(M, x) - 1)**2)

        return OM, ECC, xi2

    def BNest(self):
        """
        Use the Bhattacharyya & Nityanada method to estimate: Pb, T0, OM, ECC.
        Described in:
        Bhattacharyya & Nityanada, 2008, MNRAS, 387, Issue 1, pp. 273-278

        @return: Pb, T0, OM, ECC
        """
        Pb = self.PbEst()

        N = 50
        T0 = np.linspace(self['T0'].val, self['T0'].val+Pb, N)
        xi2 = np.zeros(N)
        OM = np.zeros(N)
        ECC = np.zeros(N)
        for ii, t in enumerate(T0):
            Peven, Podd = self.Peo(Pb=Pb, T0=t)
            OM[ii], ECC[ii], xi2[ii] = self.oe_ABC_leastsq(Peven, Podd)

        ind = np.argmin(xi2)

        return Pb, T0[ind], OM[ind], ECC[ind], xi2
