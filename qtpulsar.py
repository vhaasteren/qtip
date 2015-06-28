#!/usr/bin/python
# -*- coding: utf-8 -*-
# vim: tabstop=4:softtabstop=4:shiftwidth=4:expandtab
"""
pulsarwrapper: An adaptor class for the Qtip interface. This class will handle all
the timing package interactions. This allows Qtip to work as well with libstempo
as PINT.

"""


from __future__ import print_function
from __future__ import division

import os, sys
import pysolvepulsar as psp
import logging

# Numpy etc.
import numpy as np
import scipy.linalg as sl

# For date conversions
import jdcal        # pip install jdcal

# Use the same interface for pint and libstempo
try:
    import pint.ltinterface as lti
    ltsi = None
    have_pint = True
except ImportError:
    lti, ltsi = None, None
    have_pint = False
try:
    import libstempo as lt, libstempo.toasim as lts
    have_libstempo = True
except ImportError:
    lt, lts = None, None
    have_libstempo = False

def mjd2gcal(mjds):
    """
    Calculate the Gregorian dates from a numpy-array of MJDs

    @param mjds:    Numpy array of MJDs
    """
    nmjds = len(mjds)
    gyear = np.zeros(nmjds)
    gmonth = np.zeros(nmjds)
    gday = np.zeros(nmjds)
    gfd = np.zeros(nmjds)

    # TODO: get rid of ugly for loop. Use some zip or whatever
    for ii in range(nmjds):
        gyear[ii], gmonth[ii], gday[ii], gfd[ii] = \
                jdcal.jd2gcal(jdcal.MJD_0, mjds[ii])
    
    return gyear, gmonth, gday, gfd

# Derived PulsarSolver class, with some extra display functionality
class PSPulsar(psp.PulsarSolver):
    def __init__(self, parfile, timfile, priors=None, logfile=None,
            loglevel=logging.DEBUG, delete_prob=0.01, mP0=1.0,
            backend='libstempo'):
        """Initialize the pulsar solver

        Initialize the pulsar solver by loading a libstempo object from a
        parfile and a timfile, and by setting the priors from a prior dictionary

        :param parfile:
            Filename of a tempo2 parfile

        :param timfile:
            Filename of a tempo2 timfile

        :param priors:
            Dictionary describing the priors. priors[key] = (val, err)

        :param logfile:
            Name of the logfile

        :param loglevel:
            Level of logging

        :param delete_prob:
            Proability that an observation needs to be deleted. Can be an array,
            with per-obs probabilities as well (default: 0.01)

        :param mP0:
            How many pulse periods fit within one epoch

        :param backend:
            What timing package to use ('libstempo'/'pint')
        """
        super(PSPulsar, self).__init__(parfile, timfile, priors=priors,
                logfile=logfile, loglevel=loglevel, delete_prob=delete_prob,
                mP0=mP0, backend=backend)
        
        self.init_prediction_psr(parfile, nobs=self.nobs)

    def init_prediction_psr(self, parfile, nobs=100):
        """Initialize a fake pulsar, just for prediction purposes

        Initialize a fake pulsar, with 100 observations (for speed)

        :param parfile:
            Parfile to base the pulsar on
        """
        # TODO: Make this work with PINT!
        obstimes = np.linspace(54000, 55000, nobs)
        self.predpsr_nobs = nobs
        self.predpsr = lts.fakepulsar(parfile, obstimes, 0.1, observatory='ao')

        # Remove JUMP parameters
        for par in self.predpsr.pars(which='set'):
            if par[:4] == 'JUMP':
                self.predpsr[par].val = 0.0
                self.predpsr[par].err = 0.0
                #self.predpsr[par].set = False      # TODO: Cannot unset jumps
                self.predpsr[par].fit = False

        lts.make_ideal(self.predpsr)

    def get_mock_prediction(self, obstimes, dd):
        """Given a set of observation times, and a linear-fit results
        dictionary, return the 1-sigma prediction spread
        """
        # Create new observations for the mock pulsar
        self.predpsr.stoas[:] = obstimes
        self.predpsr.formbats()
        Mt = self.predpsr.designmatrix(fixunits=True, fixsigns=True, \
                incoffset=False)
        Mo = np.ones((self.predpsr_nobs, 1))
        Mpred = np.append(Mt, Mo, axis=1)

        # Obtain the parameter translation
        parlabels = dd['parlabels']
        parlabels_t = self.predpsr.pars(which='fit')
        parlabels_pred = list(parlabels_t) + ['PatchOffset']
        pmap = np.array([parlabels.index(par) for par in parlabels_pred])

        dpars = dd['dpars'][pmap]
        Sigma = dd['Sigma'][:,pmap][pmap,:]
        dt_r = np.dot(Mpred, dpars)
        dt_rr = np.dot(Mpred, np.dot(Sigma, Mpred.T))
        dt_stdrp = np.sqrt(np.diag(dt_rr))

        return dt_r, dt_stdrp

    def get_mock_realizations(self, obstimes, dd, ntrials=200):
        """Given a set of observation times, and a linear-fit result dictionary,
        return mock realizations so one can plot a 'Bayesiogram'"""
        # Create new observations for the mock pulsar
        self.predpsr.stoas[:] = obstimes
        self.predpsr.formbats()
        Mt = self.predpsr.designmatrix(fixunits=True, fixsigns=True, \
                incoffset=False)
        Mo = np.ones((self.predpsr_nobs, 1))
        Mpred = np.append(Mt, Mo, axis=1)

        # Obtain the parameter translation
        parlabels = dd['parlabels']
        parlabels_t = self.predpsr.pars(which='fit')
        parlabels_pred = list(parlabels_t) + ['PatchOffset']
        pmap = np.array([parlabels.index(par) for par in parlabels_pred])

        # Create the linear system
        dpars = dd['dpars'][pmap]
        Sigma = dd['Sigma'][:,pmap][pmap,:]
        cf = sl.cholesky(Sigma, lower=True)
        #realizations = np.zeros((len(obstimes), ntrials))
        xi = np.random.randn(len(Sigma), ntrials)

        # Produce the realizations
        dt_r = np.dot(Mpred, dpars)
        dt_mr = np.dot(Mpred, np.dot(cf, xi))

        #dt_rr = np.dot(Mpred, np.dot(Sigma, Mpred.T))
        #dt_stdrp = np.sqrt(np.diag(dt_rr))

        return dt_r, dt_mr

    def _orbitalphase(self):
        """Return the orbital phase

        For a binary pulsar, calculate the orbital phase. Otherwise return an
        array of zeros. (Based on the tempo2 plk plugin)
        """
        if 'T0' in self.pars(which='set'): # self['T0'].set:
            tpb = (self.toas() - self['T0'].val) / self['PB'].val
        elif 'TASC' in self.pars(which='set'): #self['TASC'].set:
            tpb = (self.toas() - self['TASC'].val) / self['PB'].val
        else:
            print("ERROR: Neither T0 nor tasc set...")
            tpb = (self.toas() - self['T0'].val) / self['PB'].val
            
        if not 'PB' in self.pars(which='set'): #self['PB'].set:
            print("WARNING: This is not a binary pulsar")
            phase = np.zeros(self.nobs)
        else:
            #if self['PB'].set:
            pbdot = self['PB'].val
            phase = tpb % 1

        return phase

    def _dayofyear(self):
        """Return the day of the year

        Return the day of the year for all the TOAs of this pulsar
        """
        gyear, gmonth, gday, gfd = mjd2gcal(self.stoas())

        mjdy = np.array([jdcal.gcal2jd(gyear[ii], 1, 0)[1] for ii in range(len(gyear))])

        return self.stoas() - mjdy

    def _year(self):
        """Return the year for all the TOAs of this pulsar
        """
        #day, jyear = self.dayandyear_old()
        gyear, gmonth, gday, gfd = mjd2gcal(self.stoas())

        # MJD of start of the year (31st Dec)
        mjdy = np.array([jdcal.gcal2jd(gyear[ii], 1, 0)[1] for ii in range(len(gfd))])
        # MJD of end of the year
        mjdy1 = np.array([jdcal.gcal2jd(gyear[ii]+1, 1, 0)[1] for ii in range(len(gfd))])

        # Day of the year
        doy = self.stoas() - mjdy
        
        return gyear + (self.stoas() - mjdy) / (mjdy1 - mjdy)

    def _siderealt(self):
        pass
        # TEMPO2 code that we can use for the sidereal time etc.

    #--- Some extra quantities we need to be able to calculate for plotting
    def pulsephase(self, cand=None, exclude_nonconnected=False):
        """Return the pulse phase, possibly excluding non-connected obsns

        Return the pulse phase. If exclude_nonconnected is True, then only
        return the values for which the patches in cand contain more than one
        toa

        Note: exclude_nonconnected will/might mess up the ordering of the
              toas

        :param cand:
            The candidate solution object

        :param exclude_nonconnected:
            If True, only return uncertainties within coherent patches with more
            than one residual
        """
        F0 = self._psr['F0'].val
        return self.residuals(cand, exclude_nonconnected) * F0

    #--- Some extra quantities we need to be able to calculate for plotting
    def orbitalphase(self, cand=None, exclude_nonconnected=False):
        """Return the orbital phase, possibly excluding non-connected obsns

        Return the orbital phase. If exclude_nonconnected is True, then only
        return the values for which the patches in cand contain more than one
        toa

        Note: exclude_nonconnected will/might mess up the ordering of the
              toas

        :param cand:
            The candidate solution object

        :param exclude_nonconnected:
            If True, only return uncertainties within coherent patches with more
            than one residual
        """
        phase = self._psr.orbitalphase()[self._isort]
        selection = self.get_canonical_selection(cand, exclude_nonconnected)
        return phase[selection]

    def dayofyear(self, cand=None, exclude_nonconnected=False):
        """Return day of the year, possibly excluding non-connected obsns

        Return the day of the year. If exclude_nonconnected is True, then only
        return the values for which the patches in cand contain more than one
        toa

        Note: exclude_nonconnected will/might mess up the ordering of the
              toas

        :param cand:
            The candidate solution object

        :param exclude_nonconnected:
            If True, only return uncertainties within coherent patches with more
            than one residual
        """
        phase = self._dayofyear()[self._isort]
        selection = self.get_canonical_selection(cand, exclude_nonconnected)
        return phase[selection]

    def year(self, cand=None, exclude_nonconnected=False):
        """Return the year, possibly excluding non-connected obsns

        Return the year. If exclude_nonconnected is True, then only
        return the values for which the patches in cand contain more than one
        toa

        Note: exclude_nonconnected will/might mess up the ordering of the
              toas

        :param cand:
            The candidate solution object

        :param exclude_nonconnected:
            If True, only return uncertainties within coherent patches with more
            than one residual
        """
        phase = self._year()[self._isort]
        selection = self.get_canonical_selection(cand, exclude_nonconnected)
        return phase[selection]

    def data_from_label(self, label, cand=None, exclude_nonconnected=False):
        """
        Given a label, return the data that corresponds to it

        @param label:   The label of which we want to obtain the data

        @return:    data, error, plotlabel
        """
        data, error, plotlabel = None, None, None

        if label == 'Residuals':
            data = self.residuals(cand=cand,
                    exclude_nonconnected=exclude_nonconnected)
            error = self.toaerrs(cand=cand,
                    exclude_nonconnected=exclude_nonconnected)
            plotlabel = r"Timing residual ($\mu$s)"
        elif label == 'mjd':
            data = self.stoas(cand=cand,
                    exclude_nonconnected=exclude_nonconnected)
            error = self.toaerrs(cand=cand,
                    exclude_nonconnected=exclude_nonconnected)
            plotlabel = r'MJD'
        elif label == 'pulse phase':
            F0 = self._psr['F0'].val
            data = self.pulsephase(cand=cand,
                    exclude_nonconnected=exclude_nonconnected)
            error = F0*self.toaerrs(cand=cand,
                    exclude_nonconnected=exclude_nonconnected)
            plotlabel = 'Pulse Phase'
        elif label == 'orbital phase':
            data = self.orbitalphase(cand=cand,
                    exclude_nonconnected=exclude_nonconnected)
            error = None
            plotlabel = 'Orbital Phase'
        elif label == 'serial':
            data = np.arange(self.nobs)
            error = None
            plotlabel = 'TOA number'
        elif label == 'day of year':
            data = self.dayofyear(cand=cand,
                    exclude_nonconnected=exclude_nonconnected)
            error = None
            plotlabel = 'Day of the year'
        elif label == 'year':
            data = self.year(cand=cand,
                    exclude_nonconnected=exclude_nonconnected)
            error = None
            plotlabel = 'Year'
        elif label == 'frequency':
            data = self.freqs(cand=cand,
                    exclude_nonconnected=exclude_nonconnected)
            error = None
            plotlabel = r"Observing frequency (MHz)"
        elif label == 'TOA error':
            data = self.toaerrs(cand=cand,
                    exclude_nonconnected=exclude_nonconnected)
            error = None
            plotlabel = "TOA uncertainty"
        elif label == 'elevation':
            data = self.elevation(cand=cand,
                    exclude_nonconnected=exclude_nonconnected)
            error = None
            plotlabel = 'Elevation'
        elif label == 'rounded MJD':
            stoas = self.stoas(cand=cand,
                    exclude_nonconnected=exclude_nonconnected)
            data = np.floor(stoas + 0.5)
            error = None
            #error = self.toaerrs(cand=cand,
            #        exclude_nonconnected=exclude_nonconnected)
            plotlabel = r'MJD'
        elif label == 'sidereal time':
            print("WARNING: parameter {0} not yet implemented".format(label))
        elif label == 'hour angle':
            print("WARNING: parameter {0} not yet implemented".format(label))
        elif label == 'para. angle':
            print("WARNING: parameter {0} not yet implemented".format(label))
        
        return data, error, plotlabel

    def siderealt(self):
        pass
        """
       else if (plot==13 || plot==14 || plot==15) 
          /* 13 = Sidereal time, 14 = hour angle, 15 = parallactic angle */
       {
          double tsid,sdd,erad,hlt,alng,hrd;
          double toblq,oblq,pc,ph;
          double siteCoord[3],ha;
          observatory *obs;
          //      printf("In here\n");
          obs = getObservatory(psr[0].obsn[iobs].telID);
          erad = sqrt(obs->x*obs->x+obs->y*obs->y+obs->z*obs->z);//height(m)
          hlt  = asin(obs->z/erad); // latitude
          alng = atan2(-obs->y,obs->x); // longitude
          hrd  = erad/(2.99792458e8*499.004786); // height (AU)
          siteCoord[0] = hrd * cos(hlt) * 499.004786; // dist from axis (lt-sec)
          siteCoord[1] = siteCoord[0]*tan(hlt); // z (lt-sec)
          siteCoord[2] = alng; // longitude

          toblq = (psr[0].obsn[iobs].sat+2400000.5-2451545.0)/36525.0;
          oblq = (((1.813e-3*toblq-5.9e-4)*toblq-4.6815e1)*toblq +84381.448)/3600.0;

          pc = cos(oblq*M_PI/180.0+psr[0].obsn[iobs].nutations[1])*psr[0].obsn[iobs].nutations[0];

          lmst2(psr[0].obsn[iobs].sat+psr[0].obsn[iobs].correctionUT1/SECDAY,0.0,&tsid,&sdd);
          tsid*=2.0*M_PI;
          /* Compute the local, true sidereal time */
          ph = tsid+pc-siteCoord[2];  
          ha = (fmod(ph,2*M_PI)-psr[0].param[param_raj].val[0])/M_PI*12;
          if (plot==13)
             x[count] = (float)fmod(ph/M_PI*12,24.0);
          else if (plot==14)
             x[count] = (float)(fmod(ha,12));
          else if (plot==15)
          {
             double cp,sqsz,cqsz,pa;
             double phi,dec;
             phi =  hlt;
             dec =  psr[0].param[param_decj].val[0];
             cp =   cos(phi);
             sqsz = cp*sin(ha*M_PI/12.0);
             cqsz = sin(phi)*cos(dec)-cp*sin(dec)*cos(ha*M_PI/12.0);
             if (sqsz==0 && cqsz==0) cqsz=1.0;
             pa=atan2(sqsz,cqsz);
             x[count] = (float)(pa*180.0/M_PI);
          }
          //      printf("Local sidereal time = %s %g %g %g %g %g\n",psr[0].obsn[iobs].fname,ph,tsid,pc,siteCoord[2],x[count]);
       }
        """

        """
// Get sidereal time
double lmst2(double mjd,double olong,double *tsid,double *tsid_der)
{
   double xlst,sdd;
   double gmst0;
   double a = 24110.54841;
   double b = 8640184.812866;
   double c = 0.093104;
   double d = -6.2e-6;
   double bprime,cprime,dprime;
   double tu0,fmjdu1,dtu,tu,seconds_per_jc,gst;
   int nmjdu1;

   nmjdu1 = (int)mjd;
   fmjdu1 = mjd - nmjdu1;

   tu0 = ((double)(nmjdu1-51545)+0.5)/3.6525e4;
   dtu  =fmjdu1/3.6525e4;
   tu = tu0+dtu;
   gmst0 = (a + tu0*(b+tu0*(c+tu0*d)))/86400.0;
   seconds_per_jc = 86400.0*36525.0;

   bprime = 1.0 + b/seconds_per_jc;
   cprime = 2.0 * c/seconds_per_jc;
   dprime = 3.0 * d/seconds_per_jc;

   sdd = bprime+tu*(cprime+tu*dprime);

   gst = gmst0 + dtu*(seconds_per_jc + b + c*(tu+tu0) + d*(tu*tu+tu*tu0+tu0*tu0))/86400;
   xlst = gst - olong/360.0;
   xlst = fortran_mod(xlst,1.0);

   if (xlst<0.0)xlst=xlst+1.0;

   *tsid = xlst;
   *tsid_der = sdd;
   return 0.0;
}
        """





class BasePulsar(object):
    """
    Base pulsar class, containing methods that do not depend on the pulsar
    timing engine.
    """

    def __init__(self):

        super(BasePulsar, self).__init__()

        self.setpriors()


    def setpriors(self):
        """Set the priors for all parameters"""
        self.isolated = ['pre-fit', 'post-fit', 'mjd', 'year', 'serial', \
            'day of year', 'frequency', 'TOA error', 'elevation', \
            'rounded MJD', 'sidereal time', 'hour angle', 'para. angle']
        self.binary = ['orbital phase']

        # Plot labels = isolated + binary
        self.plot_labels = ['pre-fit', 'post-fit', 'mjd', 'year', 'orbital phase', 'serial', \
            'day of year', 'frequency', 'TOA error', 'elevation', \
            'rounded MJD', 'sidereal time', 'hour angle', 'para. angle']

        # Some parameters we do not want to add a fitting checkbox for:
        self.nofitboxpars = ['START', 'FINISH', 'POSEPOCH', 'PEPOCH', 'DMEPOCH', \
            'EPHVER', 'TZRMJD', 'TZRFRQ', 'TRES']

        # The possible binary pulsar parameters
        self.binarypars = ['T0', 'T0_1', 'PB', 'PBDOT', 'PB_1', 'ECC', 'ECC_1', 'OM',
            'OM_1', 'A1', 'A1_1', 'OM', 'E2DOT', 'EDOT', 'KOM', 'KIN',
            'SHAPMAX', 'M2', 'MTOT', 'DR', 'DTH', 'A0', 'B0', 'BP', 'BPP', 'DTHETA',
            'SINI', 'H3', 'STIG', 'H4', 'NHARM', 'GAMMA', 'PBDOT', 'XPBDOT', 'XDOT',
            'X2DOT', 'XOMDOT', 'AFAC', 'OMDOT', 'OM2DOT', 'ORBPX', 'TASC', 'EPS1',
            'EPS1DOT', 'EPS2', 'EPS2DOT', 'TZRMJD', 'TZRFRQ', 'TSPAN', 'BPJEP_0',
            'BPJEP_1', 'BPJPH_0', 'BPJPH_1', 'BPJA1_0', 'BPJA1_1', 'BPJEC_0',
            'BPJEC_1', 'BPJOM_0', 'BPJOM_1', 'BPJPB_0', 'BPJPB_1']

        self.binmodel_ids = ['BT', 'BTJ', 'BTX', 'ELL1', 'DD', 'DDK', 'DDS',
            'MSS', 'DDGR', 'T2', 'T2-PTA', 'DDH', 'ELL1H']

        # BTmodel
        self.binmodel = OrderedDict()
        self.binmodel['BT'] = OrderedDict({
            'T0': [50000.0, 60000.0],
            'PB': [0.01, 3000],
            'ECC': [0.0, 1.0],
            'PBDOT': [0.0, 1.0e-8],
            'A1': [0.0, 1.0e3],
            'XDOT': [-1.0e-12, 1.0e-12],
            'OMDOT': [0.0, 5.0],
            'OM': [0.0, 360.0],
            'GAMMA': [0.0, 1.0]      # What is the scale of this parameter??
            })
        self.binmodel['BTJ'] = OrderedDict({
            'T0': [50000.0, 60000.0],
            'PB': [0.01, 3000],
            'ECC': [0.0, 1.0],
            'PBDOT': [0.0, 1.0e-8],
            'XDOT': [-1.0e-12, 1.0e-12],
            'A1': [0.0, 1.0e3],
            'OMDOT': [0.0, 5.0],
            'OM': [0.0, 360.0],
            'GAMMA': [0.0, 1.0],      # What is the scale of this parameter??
            'BPJEP_0': [0.0, 1.0],    # ??
            'BPJEP_1': [0.0, 1.0],
            'BPJPH_0': [0.0, 1.0],
            'BPJPH_1': [0.0, 1.0],
            'BPJA1_0': [0.0, 1.0],
            'BPJA1_1': [0.0, 1.0],
            'BPJEC_0': [0.0, 1.0],
            'BPJEC_1': [0.0, 1.0],
            'BPJOM_0': [0.0, 1.0],
            'BPJOM_1': [0.0, 1.0],
            'BPJPB_0': [0.0, 1.0],
            'BPJPB_1': [0.0, 1.0]
            })
        self.binmodel['BTX'] = OrderedDict({
            'T0': [50000.0, 60000.0],
            'ECC': [0.0, 1.0],
            'XDOT': [-1.0e-12, 1.0e-12],
            'A1': [0.0, 1.0e3],
            'OMDOT': [0.0, 5.0],
            'OM': [0.0, 360.0],
            'GAMMA': [0.0, 1.0],      # What is the scale of this parameter??
            'FB0': [0.0, 1.0],    # ??
            'FB1': [0.0, 1.0],    # ??
            'FB2': [0.0, 1.0],    # ??
            'FB3': [0.0, 1.0],    # ??
            'FB4': [0.0, 1.0],    # ??
            'FB5': [0.0, 1.0],    # ??
            'FB6': [0.0, 1.0],    # ??
            'FB7': [0.0, 1.0],    # ??
            'FB8': [0.0, 1.0],    # ??
            'FB9': [0.0, 1.0]    # ??
            })
        self.binmodel['DD'] = OrderedDict({
            'SINI': [0.0, 1.0],
            'M2': [0.0, 10.0],
            'PB': [0.01, 3000],
            'OMDOT': [0.0, 5.0],
            'T0': [50000.0, 60000.0],
            'GAMMA': [0.0, 1.0],      # What is the scale of this parameter??
            'OM': [0.0, 360.0],
            'A1': [0.0, 1.0e3],
            'XDOT': [-1.0e-12, 1.0e-12],
            'PBDOT': [0.0, 1.0e-8],
            'ECC': [0.0, 1.0],
            'XPBDOT': [0.0, 1.0],        # Units???  Scale???
            'EDOT': [0.0, 1.0]        # Units???  Scale???
            })
        self.binmodel['DDS'] = OrderedDict({
            'SHAPMAX': [0.0, 1.0],          # Scale?
            'SINI': [0.0, 1.0],
            'M2': [0.0, 10.0],
            'PB': [0.01, 3000],
            'OMDOT': [0.0, 5.0],
            'T0': [50000.0, 60000.0],
            'GAMMA': [0.0, 1.0],      # What is the scale of this parameter??
            'OM': [0.0, 360.0],
            'A1': [0.0, 1.0e3],
            'XDOT': [-1.0e-12, 1.0e-12],
            'PBDOT': [0.0, 1.0e-8],
            'ECC': [0.0, 1.0],
            'XPBDOT': [0.0, 1.0],        # Units???  Scale???
            'EDOT': [0.0, 1.0]        # Units???  Scale???
            })
        self.binmodel['DDGR'] = OrderedDict({
            'T0': [50000.0, 60000.0],
            'SINI': [0.0, 1.0],
            'M2': [0.0, 10.0],
            'PB': [0.01, 3000],
            'MTOT': [0.0, 10.0],
            'OM': [0.0, 360.0],
            'XDOT': [-1.0e-12, 1.0e-12],
            'PBDOT': [0.0, 1.0e-8],
            'XPBDOT': [0.0, 1.0],        # Units???  Scale???
            'A1': [0.0, 1.0e3],
            'ECC': [0.0, 1.0],
            'EDOT': [0.0, 1.0]        # Units???  Scale???
            })
        # Do DDH, DDK, ELL1, ELL1H, MSS, T2-PTA, T2


    def orbitalphase(self):
        """
        For a binary pulsar, calculate the orbital phase. Otherwise return an
        array of zeros. (Based on the tempo2 plk plugin)
        """
        if self['T0'].set:
            tpb = (self.toas() - self['T0'].val) / self['PB'].val
        elif self['TASC'].set:
            tpb = (self.toas() - self['TASC'].val) / self['PB'].val
        else:
            print("ERROR: Neither T0 nor tasc set...")
            tpb = (self.toas() - self['T0'].val) / self['PB'].val
            
        if not self['PB'].set:
            print("WARNING: This is not a binary pulsar")
            phase = np.zeros(len(self.toas()))
        else:
            if self['PB'].set:
                pbdot = self['PB'].val
                phase = tpb % 1

        return phase

    def dayofyear(self):
        """
        Return the day of the year for all the TOAs of this pulsar
        """
        gyear, gmonth, gday, gfd = mjd2gcal(self.stoas)

        mjdy = np.array([jdcal.gcal2jd(gyear[ii], 1, 0)[1] for ii in range(len(gyear))])

        return self.stoas - mjdy

    def year(self):
        """
        Calculate the year for all the TOAs of this pulsar
        """
        #day, jyear = self.dayandyear_old()
        gyear, gmonth, gday, gfd = mjd2gcal(self.stoas)

        # MJD of start of the year (31st Dec)
        mjdy = np.array([jdcal.gcal2jd(gyear[ii], 1, 0)[1] for ii in range(len(gfd))])
        # MJD of end of the year
        mjdy1 = np.array([jdcal.gcal2jd(gyear[ii]+1, 1, 0)[1] for ii in range(len(gfd))])

        # Day of the year
        doy = self.stoas - mjdy
        
        return gyear + (self.stoas - mjdy) / (mjdy1 - mjdy)


    def siderealt(self):
        pass
        # TEMPO2 code that we can use for the sidereal time etc.
