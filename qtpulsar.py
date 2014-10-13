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
from constants import *

# For date conversions
import jdcal        # pip install jdcal

# Import libstempo and Piccard
try:
    import piccard as pic
    print("Piccard available")
except ImportError:
    pic = None
    print("Piccard not available")

try:
    import libstempo as lt
    print("Libstempo available")
except ImportError:
    lt = None
    print("Libstempo not available")

try:
    import pint.models as tm
    from pint.phase import Phase
    from pint import toa
    import pint_temp as pinttemp
    print("PINT available")
except ImportError:
    tm = None
    Phase = None
    toa = None
    tempo2_utils = None
    print("PINT not available")



class LTPulsar(object):
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



class PPulsar(object):
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

        m.read_parfile(parfile)

        print("model.as_parfile():")
        print(m.as_parfile())

        try:
            planet_ephems = m.PLANET_SHAPIRO.value
        except AttributeError:
            planet_ephems = False

        t0 = time.time()
        t = toa.get_TOAs(timfile)
        time_toa = time.time() - t0

        sys.stderr.write("Read/corrected TOAs in %.3f sec\n" % time_toa)

        mjds = t.get_mjds()
        d_tdbs = numpy.array([x.tdb.delta_tdb_tt for x in t.table['mjd']])
        errs = t.get_errors()
        resids = numpy.zeros_like(mjds)
        ss_roemer = numpy.zeros_like(mjds)
        ss_shapiro = numpy.zeros_like(mjds)

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
        resids_us = resids / float(m.F0.value) * 1e6
        sys.stderr.write("RMS PINT residuals are %.3f us\n" % resids_us.std())

        # Get some general2 stuff
        tempo2_vals = tempo2_utils.general2(parfile, timfile,
                                            ['tt2tb', 'roemer', 'post_phase',
                                             'shapiro', 'shapiroJ'])
        t2_resids = tempo2_vals['post_phase'] / float(m.F0.value) * 1e6
        diff_t2 = resids_us - t2_resids
        diff_t2 -= diff_t2.mean()

        # run tempo1 also, if the tempo_utils module is available
        try:
            import tempo_utils
            t1_toas = tempo_utils.read_toa_file(timfile)
            tempo_utils.run_tempo(t1_toas, t1_parfile)
            t1_resids = t1_toas.get_resids(units='phase') / float(m.F0.value) * 1e6
            diff_t1 = resids_us - t1_resids
            diff_t1 -= diff_t1.mean()

            diff_t2_t1 = t2_resids - t1_resids
            diff_t2_t1 -= diff_t2_t1.mean()
        except:
            pass




        if testpulsar:
            os.remove(parfilename)
            os.remove(timfilename)

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

