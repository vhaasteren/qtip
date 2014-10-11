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

# Import libstempo and Piccard
try:
    import piccard as pic
except ImportError:
    pic is None

try:
    import libstempo as lt
except ImportError:
    lt = None


class APulsar(object):
    """
    Abstract pulsar class. For now only uses libstempo, but functionality will
    be delegated to derived classes when implementing PINT/piccard/PAL
    """

    def __init__(self, parfile, timfile):
        """
        Initialize the pulsar object
        """
        
        _psr = lt.tempopulsar(parfile, timfile)
        _interface = "libstempo"

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

    @property.setter
    def vals(self, values):
        self._psr.fitvals = values

    @property
    def fitvals(self):
        """Returns (or sets from a sequence) a numpy longdouble vector of values of all parameters that are fitted."""
        return self._psr.fitvals

    @property.setter
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

    @property.setter
    def fiterrs(self, values):
        self._psr.fiterrs = values

    @property
    def setvals(self):
        """Returns (or sets from a sequence) a numpy longdouble vector of values of all parameters that have been set."""
        return self._psr.setvals

    @property.setter
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

    @property.setter
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
    def freqs(self):
        """ Observing frequencies """
        return self._psr.freqs

    @property
    def freqsSSB(self):
        """ Observing frequencies """
        return self._psr.freqsSSB

    def residuals(self, updatebats=True, formresiduals=True):
        return self._psr.residuals(updatebats, formresiduals)

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


    @property
