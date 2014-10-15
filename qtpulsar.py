#!/usr/bin/python
# -*- coding: utf-8 -*-
# vim: tabstop=4:softtabstop=4:shiftwidth=4:expandtab
"""
qtpulsar: An adaptor class for the Qtip interface. This class will handle all
the timing package interactions. This allows Qtip to work as well with libstempo
as PINT

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
from constants import J1744_parfile, J1744_timfile

# For date conversions
import jdcal        # pip install jdcal

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
    import pint_temp
    print("PINT available")
    have_pint = True
except ImportError:
    tm = None
    Phase = None
    toa = None
    pint_temp = None
    print("PINT not available")
    have_pint = False

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

class BasePulsar(object):
    """
    Base pulsar class, containing methods that do not depend on the pulsar
    timing engine.
    """

    def __init__(self):
        self.isolated = ['pre-fit', 'post-fit', 'date', 'sidereal', \
            'day of year', 'frequency', 'TOA error', 'year', 'elevation', \
            'rounded MJD', 'sidereal time', 'hour angle', 'para. angle']
        self.binary = ['orbital phase']

        # Plot labels = isolated + binary
        self.plot_labels = ['pre-fit', 'post-fit', 'date', 'orbital phase', 'sidereal', \
            'day of year', 'frequency', 'TOA error', 'year', 'elevation', \
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
        Calculate the day of the year for all the TOAs of this pulsar
        """
        day, jyear = self.dayandyear()

        return day

    @property
    def year(self):
        """
        Calculate the year for all the TOAs of this pulsar
        """
        day, jyear = self.dayandyear()

        return jyear

    def dayandyear(self):
        """
        Calculate the day of the year, and the year
        """
        # Adjusted from plk_plug.C
        jd = self.toas + 2400000.5
        ijd = np.round(jd)
        fjd = (jd + 0.5) - ijd

        b = np.zeros(len(ijd))
        for ii, eijd in enumerate(ijd):
            if eijd > 2299160:
                a = np.floor((eijd-1867216.25)/36524.25)
                b[ii] = eijd + 1 + a - np.floor(a / 4)
            else:
                b[ii] = eijd

        c = b + 1524

        d = np.floor((c - 122.1)/365.25)
        e = np.floor(365.25*d)
        g = np.floor((c-e)/30.6001)
        day = c-e+fjd-np.floor(30.6001*g)

        month = np.zeros(len(g))
        year = np.zeros(len(g))
        for ii, eg in enumerate(g):
            if eg < 13.5:
                month[ii] = eg - 1
            else:
                month[ii] = eg - 13

            if month[ii] > 2.5:
                year[ii] = d[ii] - 4716
            else:
                year[ii] = d[ii] - 4715
            
        jyear = np.zeros(len(year))
        jmonth = np.zeros(len(day))
        jday = np.zeros(len(day))
        mjday = np.zeros(len(day))
        doy = np.zeros(len(day))
        for ii in range(len(year)):
            # Gregorian calendar to Julian calendar
            # slaCalyd(year, month, (int)day, &retYr, &retDay, &stat);
            (temp, mjday[ii]) = jdcal.gcal2jd(year[ii], month[ii], day[ii])

            (jyear[ii], jmonth[ii], jday[ii], temp) = jdcal.jd2jcal(*jdcal.gcal2jd(year[ii], month[ii], day[ii]))

            (jyear[ii], jm, jd, temp) = jdcal.jd2jcal(temp, mjday[ii])
            (temp, doy[ii]) = jdcal.jcal2jd(jyear[ii], 1, 1)
            doy[ii] += 1
            
        # Not ok yet...
        print("WARNING: jdcal not used well yet...")

        return day, jyear

    @property
    def elevation(self):
        # Need to have observatory implemented in XPulsar
        """
            observatory *obs;
            double source_elevation;

            obs = getObservatory(psr[0].obsn[iobs].telID);
            // get source elevation neglecting proper motion
            source_elevation = asin(dotproduct(psr[0].obsn[iobs].zenith,
                   psr[0].posPulsar)
                / obs->height_grs80);
            x[count] = source_elevation*180.0/M_PI;
        """
        pass

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
        


    def data_from_label(self, label):
        """
        Given a label, return the data that corresponds to it

        @param label:   The label of which we want to obtain the data

        @return:    data, error, plotlabel
        """
        data, error, plotlabel = None, None, None

        if label == 'pre-fit':
            data = self.prefitresiduals * 1e6
            error = self.toaerr * 1e6
            plotlabel = r"Pre-fit residual ($\mu$s)"
        elif label == 'post-fit':
            data = self.residuals * 1e6
            error = self.toaerr * 1e6
            plotlabel = r"Post-fit residual ($\mu$s)"
        elif label == 'date':
            data = self.toas
            error = self.toaerr * 1e6
            plotlabel = r'MJD'
        elif label == 'orbital phase':
            data = self.orbitalphase
            error = None
            plotlabel = 'Orbital Phase'
        elif label == 'sidereal':
            print("WARNING: parameter {0} not yet implemented".format(label))
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
            data = np.zeros(len(self.toas))
            error = None
            plotlabel = 'Elevation'
            print("WARNING: parameter {0} not yet implemented".format(label))
        elif label == 'rounded MJD':
            print("WARNING: parameter {0} not yet implemented".format(label))
        elif label == 'sidereal time':
            print("WARNING: parameter {0} not yet implemented".format(label))
        elif label == 'hour angle':
            print("WARNING: parameter {0} not yet implemented".format(label))
        elif label == 'para. angle':
            print("WARNING: parameter {0} not yet implemented".format(label))
        
        return data, error, plotlabel



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

            self._psr = lt.tempopulsar(parfilename, timfilename)

            os.remove(parfilename)
            os.remove(timfilename)
        elif parfile is not None and timfile is not None:
            self._psr = lt.tempopulsar(parfile, timfile)
        else:
            raise ValueError("No valid pulsar to load")

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
        raise NotImplemented("Not done")
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
    def residuals(self, updatebats=True, formresiduals=True):
        return self._psr.residuals(updatebats, formresiduals)

    @property
    def prefitresiduals(self):
        return self._psr.prefit.residuals

    def designmatrix(self, updatebats=True, fixunits=False):
        return self._psr.designmatrix(updatebats, fixunits)

    def fit(self, iters=1):
        self._psr.fit(iters)

    def chisq(self):
        return self._psr.chisq()

    def rd_hms(self):
        return self._psr.rd_hms()

    def savepar(self, parfile):
        self._psr.savepar(parfile)

    def savetim(self, timfile):
        self._psr(timfile)



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
            parfile.write(J1744_parfile)
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

        sys.stderr.write("Read/corrected TOAs in %.3f sec\n" % time_toa)

        self._mjds = t.get_mjds()
        d_tdbs = np.array([x.tdb.delta_tdb_tt for x in t.table['mjd']])
        self._toaerrs = t.get_errors()
        resids = np.zeros_like(self._mjds)
        ss_roemer = np.zeros_like(self._mjds)
        ss_shapiro = np.zeros_like(self._mjds)

        sys.stderr.write("Computing residuals...\n")
        t0 = time.time()
        for ii, tt in enumerate(t.table):
            p = m.phase(tt)
            resids[ii] = p.frac
            ss_roemer[ii] = m.solar_system_geometric_delay(tt)
            ss_shapiro[ii] = m.solar_system_shapiro_delay(tt)

        time_phase = time.time() - t0
        sys.stderr.write("Computed phases in %.3f sec\n" % time_phase)

        # resids in (approximate) us:
        self._resids_us = resids / float(m.F0.value) * 1e6
        sys.stderr.write("RMS PINT residuals are %.3f us\n" % self._resids_us.std())


        if testpulsar:
            os.remove(parfilename)
            os.remove(timfilename)


    @property
    def name(self):
        #return self._psr.name
        return "J0000+0000"

    def __getitem__(self, key):
        #return self._psr[key]
        return 0.0

    def __contains__(self, key):
        #return key in self._psr
        return False

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

