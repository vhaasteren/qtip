#!/usr/bin/python
# -*- coding: utf-8 -*-
# vim: tabstop=4:softtabstop=4:shiftwidth=4:expandtab
"""
qtpulsar: An adaptor class for the Qtip interface. This class will handle all
the timing package interactions. This allows Qtip to work as well with libstempo
as PINT.

"""


from __future__ import print_function
from __future__ import division
import os, sys

# Importing all the stuff for the IPython console widget
from IPython.qt.console.rich_ipython_widget import RichIPythonWidget
from IPython.qt.inprocess import QtInProcessKernelManager
from IPython.lib import guisupport

from PyQt4 import QtGui, QtCore

# Importing all the stuff for the matplotlib widget
import matplotlib
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure

# Numpy etc.
import numpy as np
import time
import tempfile
from constants import J1744_parfile, J1744_timfile, J1744_parfile_basic 

# For date conversions
import jdcal        # pip install jdcal

try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

# Import libstempo and Piccard
try:
    import piccard as pic
    print("Piccard available")
    have_piccard = True
except ImportError:
    pic = None
    print("Piccard not available")
    have_piccard = False

try:
    import libstempo as lt
    print("Libstempo available")
    have_libstempo = True
except ImportError:
    lt = None
    print("Libstempo not available")
    have_libstempo = False

try:
    import pint.models as tm
    from pint.phase import Phase
    from pint import toa
    #import pint_temp
    print("PINT available")
    have_pint = True
except ImportError:
    tm = None
    Phase = None
    toa = None
    #pint_temp = None
    print("PINT not available")
    have_pint = False


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

def get_engine(trypint=True):
    """
    Return a working engine

    @param trypint: If True, give priority to pint
    """
    if not trypint and have_libstempo:
        return 'libstempo', 'LTPulsar'
    elif have_pint:
        return 'pint', 'PPulsar'
    elif have_libstempo:
        return 'libstempo', 'LTPulsar'
    elif have_piccard:
        raise NotImplemented("Piccard pulsars not yet implemented")
    else:
        raise NotImplemented("Other pulsars not yet implemented")

class tempopar:
    """
    Similar to the parameter class defined in libstempo, this class gives a nice
    interface to the timing model parameters
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


class BasePulsar(object):
    """
    Base pulsar class, containing methods that do not depend on the pulsar
    timing engine.
    """

    def __init__(self):
        self.isolated = ['pre-fit', 'post-fit', 'mjd', 'year', 'serial', \
            'day of year', 'frequency', 'TOA error', 'elevation', \
            'rounded MJD', 'sidereal time', 'hour angle', 'para. angle']
        self.binary = ['orbital phase']

        # Plot labels = isolated + binary
        self.plot_labels = ['pre-fit', 'post-fit', 'mjd', 'year', 'orbital phase', 'serial', \
            'day of year', 'frequency', 'TOA error', 'elevation', \
            'rounded MJD', 'sidereal time', 'hour angle', 'para. angle']

    @property
    def orbitalphase(self):
        """
        For a binary pulsar, calculate the orbital phase. Otherwise return an
        array of zeros. (Based on the tempo2 plk plugin)
        """
        if self['T0'].set:
            tpb = (self.toas - self['T0'].val) / self['PB'].val
        elif self['TASC'].set:
            tpb = (self.toas - self['TASC'].val) / self['PB'].val
        else:
            print("ERROR: Neither T0 nor tasc set...")
            tpb = (self.toas - self['T0'].val) / self['PB'].val
            
        if not self['PB'].set:
            print("WARNING: This is not a binary pulsar")
            phase = np.zeros(len(self.toas))
        else:
            if self['PB'].set:
                pbdot = self['PB'].val
                phase = tpb % 1

        return phase

    @property
    def dayofyear(self):
        """
        Return the day of the year for all the TOAs of this pulsar
        """
        gyear, gmonth, gday, gfd = mjd2gcal(self.stoas)

        mjdy = np.array([jdcal.gcal2jd(gyear[ii], 1, 0)[1] for ii in range(len(gyear))])

        return self.stoas - mjdy

    @property
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


    @property
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
        


    def data_from_label(self, label):
        """
        Given a label, return the data that corresponds to it

        @param label:   The label of which we want to obtain the data

        @return:    data, error, plotlabel
        """
        data, error, plotlabel = None, None, None

        if label == 'pre-fit':
            data = self.prefitresiduals * 1e6
            error = self.toaerrs
            plotlabel = r"Pre-fit residual ($\mu$s)"
        elif label == 'post-fit':
            data = self.residuals * 1e6
            error = self.toaerrs
            plotlabel = r"Post-fit residual ($\mu$s)"
        elif label == 'mjd':
            data = self.stoas
            error = self.toaerrs * 1e-6
            plotlabel = r'MJD'
        elif label == 'orbital phase':
            data = self.orbitalphase
            error = None
            plotlabel = 'Orbital Phase'
        elif label == 'serial':
            data = np.arange(len(self.stoas))
            error = None
            plotlabel = 'TOA number'
        elif label == 'day of year':
            data = self.dayofyear
            error = None
            plotlabel = 'Day of the year'
        elif label == 'year':
            data = self.year
            error = None
            plotlabel = 'Year'
        elif label == 'frequency':
            data = self.freqs
            error = None
            plotlabel = r"Observing frequency (MHz)"
        elif label == 'TOA error':
            data = self.toaerrs
            error = None
            plotlabel = "TOA uncertainty"
        elif label == 'elevation':
            data = self.elevation
            error = None
            plotlabel = 'Elevation'
        elif label == 'rounded MJD':
            # TODO: Do we floor, or round like this?
            data = np.floor(self.stoas + 0.5)
            error = self.toaerrs * 1e-6
            plotlabel = r'MJD'
        elif label == 'sidereal time':
            print("WARNING: parameter {0} not yet implemented".format(label))
        elif label == 'hour angle':
            print("WARNING: parameter {0} not yet implemented".format(label))
        elif label == 'para. angle':
            print("WARNING: parameter {0} not yet implemented".format(label))
        
        return data, error, plotlabel

    def mask(self, mtype='plot', flagID=None, flagVal=None):
        """
        Returns a mask of TOAs, depending on what is requestion by mtype

        @param mtype:   What kind of mask is requested. (plot, deleted, range)
        @param flagID:  If set, only give mask for a given flag (+flagVal)
        @param flagVal: If set, only give mask for a given flag (+flagID)
        """
        msk = np.ones(len(self.stoas), dtype=np.bool)
        if mtype=='range':
            msk = np.ones(len(self.stoas), dtype=np.bool)
            if self['START'].set and self['START'].fit:
                msk[self.stoas < self['START'].val] = False
            if self['FINISH'].set and self['FINISH'].fit:
                msk[self.stoas > self['FINISH'].val] = False
        elif mtype=='deleted':
            msk = self.deleted
        elif mtype=='noplot':
            msk = self.deleted
            if self['START'].set and self['START'].fit:
                msk[self.stoas < self['START'].val] = True
            if self['FINISH'].set and self['FINISH'].fit:
                msk[self.stoas > self['FINISH'].val] = True
        elif mtype=='plot':
            msk = np.logical_not(self.deleted)
            if self['START'].set and self['START'].fit:
                msk[self.stoas < self['START'].val] = False
            if self['FINISH'].set and self['FINISH'].fit:
                msk[self.stoas > self['FINISH'].val] = False

        return msk


class LTPulsar(BasePulsar):
    """
    Abstract pulsar class. For now only uses libstempo, but functionality will
    be delegated to derived classes when implementing
    PINT/piccard/PAL/enterprise/whatever
    """

    def __init__(self, parfile=None, timfile=None, testpulsar=False):
        """
        Initialize the pulsar object

        @param parfile:     Filename of par file
        @param timfile:     Filename of tim file
        @param testpulsar:  If true, load J1744 test pulsar
        """

        self._interface = "libstempo"
        
        if testpulsar:
            # Write a test-pulsar, and open that for testing
            parfilename = tempfile.mktemp()
            timfilename = tempfile.mktemp()
            parfile = open(parfilename, 'w')
            timfile = open(timfilename, 'w')
            parfile.write(J1744_parfile)
            timfile.write(J1744_timfile)
            parfile.close()
            timfile.close()

            self._psr = lt.tempopulsar(parfilename, timfilename, dofit=False)

            os.remove(parfilename)
            os.remove(timfilename)
        elif parfile is not None and timfile is not None:
            self._psr = lt.tempopulsar(parfile, timfile, dofit=False)
        else:
            raise ValueError("No valid pulsar to load")

        # Some parameters we do not want to add a fitting checkbox for:
        self.nofitboxpars = ['START', 'FINISH', 'POSEPOCH', 'PEPOCH', 'DMEPOCH', \
            'EPHVER']

        # The possible binary pulsar parameters
        self.binarypars = ['T0', 'T0_1', 'PB', 'PBDOT', 'PB_1', 'ECC', 'ECC_1', 'OM',
            'OM_1', 'A1', 'A1_1', 'OM', 'OM_1', 'E2DOT', 'EDOT', 'KOM', 'KIN',
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

    @property
    def name(self):
        return self._psr.name

    def __getitem__(self, key):
        return self._psr[key]

    def __contains__(self, key):
        return key in self._psr

    @property
    def pars(self):
        """Returns tuple of names of parameters that are fitted (deprecated, use fitpars)."""
        return self._psr.fitpars

    @property
    def fitpars(self):
        """Returns tuple of names of parameters that are fitted."""
        return self._psr.fitpars

    @property
    def setpars(self):
        """Returns tuple of names of parameters that have been set."""
        return self._psr.setpars

    @property
    def allpars(self):
        """Returns tuple of names of all tempo2 parameters (whether set or unset, fit or not fit)."""
        return self._psr.allpars

    @property
    def vals(self):
        """Returns (or sets from a sequence) a numpy longdouble vector of values of all parameters that are fitted (deprecated, use fitvals)."""
        return self._psr.vals

    @vals.setter
    def vals(self, values):
        self._psr.fitvals = values

    @property
    def fitvals(self):
        """Returns (or sets from a sequence) a numpy longdouble vector of values of all parameters that are fitted."""
        return self._psr.fitvals

    @fitvals.setter
    def fitvals(self, values):
        self._psr.fitvals = values

    @property
    def errs(self):
        """Returns a numpy longdouble vector of errors of all parameters that are fitted."""
        return self._psr.fiterrs

    @property
    def fiterrs(self):
        """Returns a numpy longdouble vector of errors of all parameters that are fitted."""
        return self._psr.fiterrs

    @fiterrs.setter
    def fiterrs(self, values):
        self._psr.fiterrs = values

    @property
    def setvals(self):
        """Returns (or sets from a sequence) a numpy longdouble vector of values of all parameters that have been set."""
        return self._psr.setvals

    @setvals.setter
    def setvals(self, values):
        self._psr.setvals = values

    @property
    def seterrs(self):
        """Returns a numpy longdouble vector of errors of all parameters that have been set."""
        return self._psr.seterrs

    # the best way to access prefit pars would be through the same interface:
    # psr.prefit['parname'].val, psr.prefit['parname'].err, perhaps even psr.prefit.cols
    # since the prefit values don't change, it's OK for psr.prefit to be a static attribute

    @property
    def binarymodel(self):
        return self._psr.binaryModel

    @property
    def ndim(self):
        return self._psr.ndim

    @property
    def deleted(self):
        return self._psr.deleted

    @deleted.setter
    def deleted(self, values):
        self._psr.deleted = values

    @property
    def toas(self):
        """ Barycentric arrival times """
        return self._psr.toas()

    @property
    def stoas(self):
        """ Site arrival times """
        return self._psr.stoas

    @property
    def toaerrs(self):
        """ TOA uncertainties """
        return self._psr.toaerrs

    @property
    def freqs(self):
        """ Observing frequencies """
        return self._psr.freqs

    @property
    def freqsSSB(self):
        """ Observing frequencies """
        return self._psr.freqsSSB

    @property
    def elevation(self):
        """Source elevation"""
        return self._psr.elevation()

    @property
    def residuals(self, updatebats=True, formresiduals=True):
        return self._psr.residuals(updatebats, formresiduals)

    @property
    def prefitresiduals(self):
        return self._psr.prefit.residuals

    def designmatrix(self, updatebats=True, fixunits=False):
        return self._psr.designmatrix(updatebats, fixunits)

    def fit(self, iters=1):
        """
        Perform a fit with tempo2
        """
        # TODO: Figure out why the START and FINISH parameters get modified, and
        # then fix this in libstempo?
        if self['START'].set:
            start = self['START'].val
        else:
            start = None

        if self['FINISH'].set:
            finish = self['FINISH'].val
        else:
            finish = None

        # Perform the fit
        self._psr.fit(iters)

        if start is not None:
            self['START'].val = start

        if finish is not None:
            self['FINISH'].val = finish

    def chisq(self):
        return self._psr.chisq()

    def rd_hms(self):
        return self._psr.rd_hms()

    def savepar(self, parfile):
        self._psr.savepar(parfile)

    def savetim(self, timfile):
        self._psr(timfile)

    def phasejumps(self):
        return self._psr.phasejumps()

    def add_phasejump(self, mjd, phasejump):
        self._psr.add_phasejump(mjd, phasejump)

    def remove_phasejumps(self):
        self._psr.remove_phasejumps()

    @property
    def nphasejumps(self):
        return self._psr.nphasejumps



class PPulsar(BasePulsar):
    """
    Abstract pulsar class. For now only uses PINT
    """

    def __init__(self, parfile=None, timfile=None, testpulsar=False):
        """
        Initialize the pulsar object

        @param parfile:     Filename of par file
        @param timfile:     Filename of tim file
        @param testpulsar:  If true, load J1744 test pulsar
        """

        # Create a timing-model
        self._interface = "pint"
        m = tm.StandardTimingModel()
        
        if testpulsar:
            # Write a test-pulsar, and open that for testing
            parfilename = tempfile.mktemp()
            timfilename = tempfile.mktemp()
            parfile = open(parfilename, 'w')
            timfile = open(timfilename, 'w')
            parfile.write(J1744_parfile_basic)
            timfile.write(J1744_timfile)
            parfile.close()
            timfile.close()
        elif parfile is not None and timfile is not None:
            pass
        else:
            raise ValueError("No valid pulsar to load")

        # We have a par/tim file. Read them in!
        m.read_parfile(parfilename)

        print("model.as_parfile():")
        print(m.as_parfile())

        try:
            planet_ephems = m.PLANET_SHAPIRO.value
        except AttributeError:
            planet_ephems = False

        t0 = time.time()
        t = toa.get_TOAs(timfilename)
        time_toa = time.time() - t0
        t.print_summary()

        sys.stderr.write("Read/corrected TOAs in %.3f sec\n" % time_toa)

        self._mjds = t.get_mjds()
        #d_tdbs = np.array([x.tdb.delta_tdb_tt for x in t.table['mjd']])
        self._toaerrs = t.get_errors()
        resids = np.zeros_like(self._mjds)
        #ss_roemer = np.zeros_like(self._mjds)
        #ss_shapiro = np.zeros_like(self._mjds)

        sys.stderr.write("Computing residuals...\n")
        t0 = time.time()
        phases = m.phase(t.table)
        resids = phases.frac

        #for ii, tt in enumerate(t.table):
        #    p = m.phase(tt)
        #    resids[ii] = p.frac
        #    ss_roemer[ii] = m.solar_system_geometric_delay(tt)
        #    ss_shapiro[ii] = m.solar_system_shapiro_delay(tt)

        time_phase = time.time() - t0
        sys.stderr.write("Computed phases in %.3f sec\n" % time_phase)

        # resids in (approximate) us:
        self._resids_us = resids / float(m.F0.value) * 1e6
        sys.stderr.write("RMS PINT residuals are %.3f us\n" % self._resids_us.std())

        # Create a dictionary of the fitting parameters
        self.pardict = OrderedDict()
        self.pardict['START'] = tempopar('START')
        self.pardict['FINISH'] = tempopar('FINISH')
        self.pardict['RAJ'] = tempopar('RAJ')
        self.pardict['DECJ'] = tempopar('DECJ')
        self.pardict['PMRA'] = tempopar('PMRA')
        self.pardict['PMDEC'] = tempopar('PMDEC')
        self.pardict['F0'] = tempopar('F0')
        self.pardict['F1'] = tempopar('F1')


        if testpulsar:
            os.remove(parfilename)
            os.remove(timfilename)


    @property
    def name(self):
        #return self._psr.name
        return "J0000+0000"

    def __getitem__(self, key):
        #return self._psr[key]
        return self.pardict[key]

    def __contains__(self, key):
        #return key in self._psr
        return key in self.pardict

    @property
    def pars(self):
        """Returns tuple of names of parameters that are fitted (deprecated, use fitpars)."""
        #return self._psr.fitpars
        return ('none')

    @property
    def fitpars(self):
        """Returns tuple of names of parameters that are fitted."""
        #return self._psr.fitpars
        return ('F0', 'F1')

    @property
    def setpars(self):
        """Returns tuple of names of parameters that have been set."""
        #return self._psr.setpars
        return ('F0', 'F1')

    @property
    def allpars(self):
        """Returns tuple of names of all tempo2 parameters (whether set or unset, fit or not fit)."""
        #return self._psr.allpars
        return ('F0', 'F1')

    @property
    def vals(self):
        """Returns (or sets from a sequence) a numpy longdouble vector of values of all parameters that are fitted (deprecated, use fitvals)."""
        #return self._psr.vals
        return np.array([0.0, 0.0])

    @vals.setter
    def vals(self, values):
        #self._psr.fitvals = values
        pass

    @property
    def fitvals(self):
        """Returns (or sets from a sequence) a numpy longdouble vector of values of all parameters that are fitted."""
        #return self._psr.fitvals
        return np.array([0.0, 0.0])

    @fitvals.setter
    def fitvals(self, values):
        #self._psr.fitvals = values
        pass

    @property
    def errs(self):
        """Returns a numpy longdouble vector of errors of all parameters that are fitted."""
        #return self._psr.fiterrs
        return np.array([0.0, 0.0])

    @property
    def fiterrs(self):
        """Returns a numpy longdouble vector of errors of all parameters that are fitted."""
        #return self._psr.fiterrs
        return np.array([0.0, 0.0])

    @fiterrs.setter
    def fiterrs(self, values):
        #self._psr.fiterrs = values
        pass

    @property
    def setvals(self):
        """Returns (or sets from a sequence) a numpy longdouble vector of values of all parameters that have been set."""
        #return self._psr.setvals
        return np.array([0.0, 0.0])

    @setvals.setter
    def setvals(self, values):
        #self._psr.setvals = values
        pass

    @property
    def seterrs(self):
        """Returns a numpy longdouble vector of errors of all parameters that have been set."""
        #return self._psr.seterrs
        return np.array([0.0, 0.0])

    # the best way to access prefit pars would be through the same interface:
    # psr.prefit['parname'].val, psr.prefit['parname'].err, perhaps even psr.prefit.cols
    # since the prefit values don't change, it's OK for psr.prefit to be a static attribute

    @property
    def binarymodel(self):
        #return self._psr.binaryModel
        return 'single'

    @property
    def ndim(self):
        #return self._psr.ndim
        return 2

    @property
    def deleted(self):
        #return self._psr.deleted
        return np.zeros(len(self._mjds), dtype=np.bool)

    @deleted.setter
    def deleted(self, values):
        #self._psr.deleted = values
        pass

    @property
    def toas(self):
        """ Barycentric arrival times """
        #return self._psr.toas()
        return self._mjds

    @property
    def stoas(self):
        """ Site arrival times """
        raise NotImplemented("Not done")
        #return self._psr.stoas
        return self._mjds

    @property
    def toaerrs(self):
        """ TOA uncertainties """
        return self._toaerrs

    @property
    def freqs(self):
        """ Observing frequencies """
        #return self._psr.freqs
        return np.zeros(len(self._mjds), dtype=np.double)

    @property
    def freqsSSB(self):
        """ Observing frequencies """
        #return self._psr.freqsSSB
        return np.zeros(len(self._mjds), dtype=np.double)

    @property
    def residuals(self, updatebats=True, formresiduals=True):
        #return self._psr.residuals(updatebats, formresiduals)
        return self._resids_us*1e-6

    @property
    def prefitresiduals(self):
        #return self._psr.prefit.residuals
        return self._resids_us*1e-6

    def designmatrix(self, updatebats=True, fixunits=False):
        #return self._psr.designmatrix(updatebats, fixunits)
        raise NotImplemented("Not done")
        return None

    def fit(self, iters=1):
        #self._psr.fit(iters)
        raise NotImplemented("Not done")

    def chisq(self):
        raise NotImplemented("Not done")
        #return self._psr.chisq()
        return 0.0

    def rd_hms(self):
        raise NotImplemented("Not done")
        #return self._psr.rd_hms()
        return None

    def savepar(self, parfile):
        #self._psr.savepar(parfile)
        pass

    def savetim(self, timfile):
        #self._psr(timfile)
        pass

