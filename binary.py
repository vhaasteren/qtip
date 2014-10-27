#!/usr/bin/python
# -*- coding: utf-8 -*-
# vim: tabstop=4:softtabstop=4:shiftwidth=4:expandtab

"""
Binary: Qt interactive interface to explore binary parameters


"""

from __future__ import print_function
from __future__ import division
import os, sys

# Importing all the stuff for the IPython console widget
from PyQt4 import QtGui, QtCore

# Importing all the stuff for the matplotlib widget
import matplotlib
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure

# Numpy etc.
import numpy as np
import time
import copy

# For date conversions
import jdcal        # pip install jdcal

import constants
import qtpulsar as qp
import tempfile





###############################################################################
######################## Direct copy from fitorbit.py #########################
###############################################################################

import math 
#import slalib
#from pyslalib import *         # For Degrees/Arcmin/Arcsec to Radians, and Hour/Min/Sec to Radians
from scipy.optimize import leastsq

import fitorbit_parfile as parfile
from utils import eccentric_anomaly

DEG2RAD    = float('1.7453292519943295769236907684886127134428718885417e-2')
RAD2DEG    = float('57.295779513082320876798154814105170332405472466564')
C           = float('2.99792458e8')

PARAMS = ['RA', 'DEC', 'P0', 'P1', 'PEPOCH', 'PB', 'ECC', 'A1', 'T0', 'OM']


# Declarations for MENU
"""
entries = (
  ( "FileMenu", None, "File" ),               # name, stock id, label
  ( "PreferencesMenu", None, "Preferences" ), # name, stock id, label
  ( "HelpMenu", None, "Help" ),               # name, stock id, label
  ( "Save", gtk.STOCK_SAVE, "_Save","<control>S", "Save current file", activate_action ),
  ( "SaveAs", gtk.STOCK_SAVE, "Save _As...","<control>A", "Save to a file", activate_action ),
  ( "Quit", gtk.STOCK_QUIT, "_Quit", "<control>Q", "Quit", quit  ),
  ( "About", None, "_About", "<control>H", "About", activate_action ),
  ( "Logo", "demo-gtk-logo", None, None, "GTK+", activate_action ),
)
"""


ui_info = \
'''<ui>
  <menubar name='MenuBar'>
    <menu action='FileMenu'>
      <menuitem action='Save'/>
      <menuitem action='SaveAs'/>
      <separator/>
      <menuitem action='Quit'/>
    </menu>
    <menu action='PreferencesMenu'>
    </menu>
    <menu action='HelpMenu'>
      <menuitem action='About'/>
    </menu>
  </menubar>
  <toolbar  name='ToolBar'>
    <toolitem action='Quit'/>
    <separator action='Sep1'/>
    <toolitem action='Logo'/>
  </toolbar>
</ui>'''



class Param:
    def __init__(self, is_string=False):
        self.val = 0.0
        if is_string:
            self.val = "00:00:00.0"
        self.fit = 0


# Function to calc the expected period at a time x (in MJD) given the parameters
def calc_period(x, DRA, DDEC, P0, P1, PEPOCH, PB, ECC, A1, T0, OM, RA, DEC):

    k1 = 2*np.pi*A1/(PB*86400.0*np.sqrt(1-ECC*ECC))

    # Calc easc in rad
    easc = 2*np.arctan(np.sqrt((1-ECC)/(1+ECC)) * np.tan(-OM*DEG2RAD/2))
    #print easc
    epperias = T0 - PB/360.0*(RAD2DEG * easc - RAD2DEG * ECC * np.sin(easc))
    #print x,epperias
    mean_anom = 360*(x-epperias)/PB
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
    #dv = deltav(x, RA, DEC, RA-DRA, DEC-DDEC, 2000.0)
    #print dv

    return 1000*(P0+P1*1e-15*(x-PEPOCH)*86400) * (1+k1*np.cos(DEG2RAD*(true_anom+OM)) )
    #return 1000*(P0+P1*1e-15*(x-PEPOCH)*86400) * (1+k1*np.cos(DEG2RAD*(true_anom+OM)) ) * (1-20000/C)

# Function to calc Period residual y-f(x,...)
def resid_period(param, Pobs, x, fit, fixed_values):
    """
    param : value of the M parameters to fit
    Pobs : array of the f(x) values
    x : array of the x values

    fit : Array of N parameters which indicate the M parameters to fit  
    fixed_values : values of the fixed parameters
    """

    nb_fit=0


    # DRA 
    if fit[0]!=0:
        nb_fit+=1
    DRA = 0.0 

    # DDEC
    if fit[1]!=0:
        nb_fit+=1
    DDEC = 0.0 

    # P0
    if fit[2]!=0:
        P0 = param[nb_fit]
        nb_fit+=1
    else:
        P0 = fixed_values[2]

    # P1
    if fit[3]!=0:
        P1 = param[nb_fit]
        nb_fit +=1
    else:
        P1 = fixed_values[3]

    # PEPOCH
    if fit[4]!=0:
        PEPOCH = param[nb_fit]
        nb_fit +=1
    else:
        PEPOCH = fixed_values[4] 

    # PB
    if fit[5]!=0:
        PB = param[nb_fit]
        nb_fit +=1
    else:
        PB = fixed_values[5]

    # ECC
    if fit[6]!=0:
        ECC = param[nb_fit]
        nb_fit +=1
    else:
        ECC = fixed_values[6]

    # A1
    if fit[7]!=0:
        A1 = param[nb_fit]
        nb_fit +=1
    else:
        A1 = fixed_values[7]

    # T0
    if fit[8]!=0:
        T0 = param[nb_fit]
        nb_fit +=1
    else:
        T0 = fixed_values[8]

    # A1
    if fit[9]!=0:
        OM = param[nb_fit]
        nb_fit +=1
    else:
        OM = fixed_values[9]

    # RA
    RA = fixed_values[0]
    DEC = fixed_values[1]

    return Pobs - calc_period(x, DRA, DDEC, P0, P1, PEPOCH, PB, ECC, A1, T0, OM, RA, DEC)














###############################################################################
######################## Qt code that mimicks fitorbit ########################
###############################################################################




class BinaryWidget(QtGui.QWidget):
    """
    The binary-widget window.

    @param parent:      Parent window
    """

    def __init__(self, parent=None, parfilename=None, perfilename=None, **kwargs):
        super(BinaryWidget, self).__init__(parent, **kwargs)

        self.initBin()

        self.psr = None
        self.parent = parent

    def initBin(self):
        """
        Initialize all the Widgets, and add them to the layout
        """
        self.numpars = len(PARAMS)
        self.parameterCols = 2
        self.parameterRows = int(np.ceil(self.numpars / self.parameterCols))
        cblength = 10
        inplength = 15

        self.setMinimumSize(650, 550)

        # The layout boxes and corresponding widgets
        self.fullwidgetbox = QtGui.QHBoxLayout()        # whole widget
        self.operationbox = QtGui.QVBoxLayout()         # operation sect. (left)
        self.inoutputbox = QtGui.QVBoxLayout()          # in and output (right)
        self.parameterbox = QtGui.QGridLayout()         # The model parameters

        # The binaryModel Combobox Widget
        self.binaryModelCB = QtGui.QComboBox()
        self.binaryModelCB.addItem('DD')
        self.binaryModelCB.addItem('T2')
        self.binaryModelCB.addItem('ELL')
        #self.binaryModelCB.stateChanged.connect(self.changedBinaryModel)
        self.operationbox.addWidget(self.binaryModelCB)

        # The action buttons
        self.fitButton = QtGui.QPushButton('Fit Model')
        self.fitButton.clicked.connect(self.fitModel)
        self.operationbox.addWidget(self.fitButton)

        # Checkbox for yes/no plot Model
        self.plotCheckBox = QtGui.QCheckBox('Plot Model')
        self.plotCheckBox.stateChanged.connect(self.changedPlotModel)
        self.operationbox.addWidget(self.plotCheckBox)

        # Finish the operation Widget
        self.operationbox.addStretch(1)
        self.fullwidgetbox.addLayout(self.operationbox)

        # Place the model boxes on a grid
        self.parameterbox = QtGui.QGridLayout()
        self.parameterbox.setSpacing(10)

        # Add all the parameters
        index = 0
        for ii in range(self.parameterRows):
            for jj in range(self.parameterCols):
                if index < self.numpars:
                    # Add another parameter to the grid
                    offset = jj*(cblength + inplength)

                    checkbox = QtGui.QCheckBox(PARAMS[index], parent=self)
                    self.parameterbox.addWidget(checkbox, \
                            ii, offset, 1, cblength)

                    textedit = QtGui.QTextEdit("", parent=self)
                    self.parameterbox.addWidget(textedit, \
                            ii, offset+cblength, 1, inplength)

                    # TODO: Make callback functions for when these change
                    index += 1

        # Finalize the parameter Widget
        self.inoutputbox.addLayout(self.parameterbox)

        # We are creating the Figure here, so set the color scheme appropriately
        self.setColorScheme(True)

        # Create the mpl Figure and FigCanvas objects. 
        # 5x4 inches, 100 dots-per-inch
        #
        self.binDpi = 100
        self.binFig = Figure((5.0, 4.0), dpi=self.binDpi)
        self.binCanvas = FigureCanvas(self.binFig)
        self.binCanvas.setParent(self)

        # Since we have only one plot, we can use add_axes 
        # instead of add_subplot, but then the subplot
        # configuration tool in the navigation toolbar wouldn't
        # work.
        #
        self.binAxes = self.binFig.add_subplot(111)

        # Done creating the Figure. Restore color scheme to defaults
        self.setColorScheme(False)
        
        # Call-back functions for clicking and key-press.
        self.binCanvas.mpl_connect('button_press_event', self.canvasClickEvent)
        self.binCanvas.mpl_connect('key_press_event', self.canvasKeyEvent)

        # Draw an empty graph
        self.drawSomething()

        # Create the navigation toolbar, tied to the canvas
        #
        #self.mpl_toolbar = NavigationToolbar(self.canvas, self.main_frame)

        # Add the figure to the in/output Layout/Widget
        #self.inoutputbox.addStretch(1)
        #self.inoutputbox.addWidget(self.binCanvas)
        self.fullwidgetbox.addLayout(self.inoutputbox)
        self.setLayout(self.fullwidgetbox)


    def setColorScheme(self, start=True):
        """
        Set the color scheme

        @param start:   When true, save the original scheme, and set to white
                        When False, restore the original scheme
        """
        # Obtain the Widget background color
        color = self.palette().color(QtGui.QPalette.Window)
        r, g, b = color.red(), color.green(), color.blue()
        rgbcolor = (r/255.0, g/255.0, b/255.0)

        if start:
            # Copy of 'white', because of bug in matplotlib that does not allow
            # deep copies of rcParams. Store values of matplotlib.rcParams
            self.orig_rcParams = copy.deepcopy(constants.mpl_rcParams_white)
            for key, value in self.orig_rcParams.iteritems():
                self.orig_rcParams[key] = matplotlib.rcParams[key]

            rcP = copy.deepcopy(constants.mpl_rcParams_white)
            rcP['axes.facecolor'] = rgbcolor
            rcP['figure.facecolor'] = rgbcolor
            rcP['figure.edgecolor'] = rgbcolor
            rcP['savefig.facecolor'] = rgbcolor
            rcP['savefig.edgecolor'] = rgbcolor

            for key, value in rcP.iteritems():
                matplotlib.rcParams[key] = value
        else:
            for key, value in constants.mpl_rcParams_black.iteritems():
                matplotlib.rcParams[key] = value

    def drawSomething(self):
        """
        When we don't have a pulsar yet, but we have to display something, just draw
        an empty figure
        """
        self.setColorScheme(True)
        self.binAxes.clear()
        self.binAxes.grid(True)
        self.binAxes.set_xlabel('MJD')
        self.binAxes.set_ylabel('Period ($\mu$s)')
        self.binCanvas.draw()
        self.setColorScheme(False)

    def init_param_file():
        """
        Init parameters of PARFILE
             fit_flag[] : which parameters to fit
             fit_values=[] : values of parameters
        """
        self.param = parfile.Parfile()

        # Array for LM fit
        self.fit_flag=[]
        self.fit_values=[]
        self.param2fit=[]
        self.mjds2=[]
        self.ps2=[]

        # Dict p2f for parameters to fit
        self.p2f={}
        self.label=[]
        for PARAM in PARAMS:
            if PARAM=="RA" or PARAM=="DEC":
                self.p2f[PARAM] = Param(is_string=True)
            else:        
                self.p2f[PARAM] = Param()
            self.label.append(PARAM)


        # Init self.fit to 0
        for i in range(len(self.p2f)):
            self.fit_flag.append(0)

    def openPulsar(self, perfilename=None, parfilename=None):
        """
        Open a per/par file.

        TODO: This needs to be in the kernel namespace. But for now, keep it in
              the widget
        """
        if perfilename is None or parfilename is None:
            # Write temporary files
            tperfilename = tempfile.mktemp()
            tparfilename = tempfile.mktemp()
            tperfile = open(tperfilename, 'w')
            tparfile = open(tparfilename, 'w')
            tperfile.write(constants.J1903PER)
            tparfile.write(constants.J1903EPH)
            tperfile.close()
            tparfile.close()
        else:
            tperfilename = perfilename
            tparfilename = parfilename


        # Read the ephemeris (par) file
        self.param.read(tparfilename)
        self.p2f['RA'].val = self.param.RAJ
        self.p2f['DEC'].val = self.param.DECJ
        self.p2f['P0'].val = self.param.P0
        self.p2f['P1'].val = self.param.P1/1e-15
        self.p2f['PEPOCH'].val = self.param.PEPOCH
        self.p2f['PB'].val = self.param.PB
        self.p2f['ECC'].val = self.param.ECC
        self.p2f['A1'].val = self.param.A1
        self.p2f['T0'].val = self.param.T0
        self.p2f['OM'].val = self.param.OM

        # Read the files here
        self.mjds, self.periods = np.loadtxt(tperfilename, usecols=(0,1), unpack=True)

        # if flgfreq self.periods=1.0/self.periods
        # if flgms:
        if False:
            pass
        else:
            # Give pulse period in milliseconds
            self.periods = self.periods * 1000.


        if perfilename is None or parfilename is None:
            os.remove(tperfilename)
            os.remove(tparfilename)

    def write_param_file(self):
        for PARAM in PARAMS:
            self.param.set_param(PARAM, self.p2f[PARAM].val)


    def plot_model(self, widget=None):
        """

        # Retrieve values from the query
        for i in range(len(self.p2f)):
            #print self.label[i]
            if self.label[i]=='RA':
                raj_entry = self.local_entry[i].get_text().split(':')
                ra_radian = 0.0
                if len(raj_entry) > 1:
                    # Entry in HH:MM:SS
                    (rah,ram,ras) = raj_entry
                    (ra_radian,flag) = slalib.sla_dtf2r(rah,ram,ras)
                else:
                    # Entry in radians
                    ra_radian = float(raj_entry[0])

                self.p2f[self.label[i]].val = ra_radian 

            elif self.label[i]=='DEC':
                decj_entry = self.local_entry[i].get_text().split(':')
                dec_radian = 0.0
                if len(decj_entry) > 1:
                    # Entry in HH:MM:SS
                    (dech,decm,decs) = decj_entry
                    (dec_radian,flag) = slalib.sla_dtf2r(dech,decm,decs)
                else:
                    # Entry in radians
                    dec_radian = float(decj_entry[0])

                self.p2f[self.label[i]].val = dec_radian 
            else:
                self.p2f[self.label[i]].val = float(self.local_entry[i].get_text())
                #print i, self.label[i], self.p2f[self.label[i]].val, self.local_entry[i].get_text()
          
        # Init arrays
        xs=np.linspace(min(self.mjds),max(self.mjds),2000)


        ys=calc_period(xs, 0.0, 0.0, self.p2f['P0'].val, self.p2f['P1'].val, self.p2f['PEPOCH'].val, self.p2f['PB'].val, self.p2f['ECC'].val, self.p2f['A1'].val, self.p2f['T0'].val, self.p2f['OM'].val, self.p2f['RA'].val, self.p2f['DEC'].val) 

        
        # Convert into a Numpy array
        ys=np.asarray(ys)

        # Redraw plot
        self.ax1.cla()
        self.ax1.plot(self.mjds,self.periods,'r+',ms=9)
        line, = self.ax1.plot(xs, ys)

        # Label and axis
        self.ax1.set_xlabel(self.xlabel)
        self.ax1.set_ylabel(self.ylabel)
        self.ax1.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        self.ax1.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))

        self.canvas.draw()
        """

    def changedBinaryModel(self):
        """
        Called when we change the state of the Binary Model combobox.
        """
        pass

    def changedPlotModel(self):
        """
        Called when we change the state of the PlotModel checkbox.
        """
        pass


    def fitModel(self, widget=None):
        """
        Function to perform the fit of selected parameters to the values
        """
        self.setColorScheme(True)
        self.binAxes.clear()
        self.binAxes.grid(True)

        self.binAxes.plot([1, 2], [1, 2])

        self.binCanvas.draw()
        self.setColorScheme(False)
        
        """
        # Retrieve values of parameters
        self.fit_values = []
        for i in range(len(self.p2f)):
            if self.label[i]=='RA':
                raj_entry = self.local_entry[i].get_text().split(':')
                ra_radian = 0.0
                if len(raj_entry) > 1:
                    # Entry in HH:MM:SS
                    (rah,ram,ras) = raj_entry
                    (ra_radian,flag) = slalib.sla_dtf2r(rah,ram,ras)
                else:
                    # Entry in radians
                    ra_radian = float(raj_entry[0])

                self.fit_values.append( ra_radian )

            elif self.label[i]=='DEC':
                decj_entry = self.local_entry[i].get_text().split(':')
                dec_radian = 0.0
                if len(decj_entry) > 1:
                    # Entry in HH:MM:SS
                    (dech,decm,decs) = decj_entry
                    (dec_radian,flag) = slalib.sla_dtf2r(dech,decm,decs)
                else:
                    # Entry in radians
                    dec_radian = float(decj_entry[0])

                self.fit_values.append( dec_radian )
            else:
                self.fit_values.append( float(self.local_entry[i].get_text()) )


        # Get which parameters will be fitted
        self.param2fit = []
        for i,dofit in enumerate(self.fit_flag):
            if dofit:
                self.param2fit.append( self.fit_values[i] )

        # If not parameters will be fitted, return now !
        if not self.param2fit:
            return


        # Retrieve which points to include (points in the window)
        self.ps2=[]
        self.mjds2=[]
        xmin,xmax=self.ax1.get_xlim()
        for ii, mjd in enumerate(self.mjds):
            if(xmin<mjd and mjd<xmax):
              self.mjds2.append(mjd)
              self.ps2.append(self.periods[ii])

        self.mjds2 = np.asarray(self.mjds2)
        self.ps2 = np.asarray(self.ps2)
        #print self.mjds2,self.ps2

        # Do least square fit
        print 'Input Parameters :\n',self.param2fit
        #print self.ps2, self.mjds2, self.fit_flag, self.fit_values
        plsq = leastsq(resid_period, self.param2fit, args=(self.ps2, self.mjds2, self.fit_flag, self.fit_values))
        print 'Parameters fitted :\n', plsq[0]
        #print resid_period(self.param2fit,self.ps2, self.mjds2, self.fit, self.fit_values)

        print 'chi**2 = ',np.sum(np.power(resid_period(self.param2fit,self.ps2, self.mjds2, self.fit_flag, self.fit_values),2))
        # Return new parameters values in boxes
        j=0
        for i,dofit in enumerate(self.fit_flag):
          #print i,dofit, plsq
          if dofit:
            if sum(self.fit_flag)>=1:
                self.local_entry[i].set_text(str(plsq[0][j]))
                j+=1
            else:
                self.local_entry[i].set_text(str(plsq[0]))

        # Update the plot
        self.plot_model()
        """



    def newFitParameters(self):
        """
        This function is called when we have new fitparameters

        TODO: callback not used right now
        """
        pass


    def updatePlot(self):
        """
        Update the plot/figure
        """
        if self.psr is not None:
            # Get a mask for the plotting points
            msk = self.psr.mask('plot')

            #print("Mask has {0} toas".format(np.sum(msk)))

            # Get the IDs of the X and Y axis
            #xid, yid = self.xyChoiceWidget.plotids()
            xid, yid = 'MJD', 'post-fit'

            # Retrieve the data
            x, xerr, xlabel = self.psr.data_from_label(xid)
            y, yerr, ylabel = self.psr.data_from_label(yid)

            if x is not None and y is not None and np.sum(msk) > 0:
                xp = x[msk]
                yp = y[msk]

                if yerr is not None:
                    yerrp = yerr[msk]
                else:
                    yerrp = None

                self.updatePlotL(xp, yp, yerrp, xlabel, ylabel, self.psr.name)
            else:
                raise ValueError("Nothing to plot!")


    def updatePlotL(self, x, y, yerr, xlabel, ylabel, title):
        """
        Update the plot, given all the plotting info
        """
        self.setColorScheme(True)
        self.binAxes.clear()
        self.binAxes.grid(True)

        xave = 0.5 * (np.max(x) + np.min(x))
        xmin = xave - 1.05 * (xave - np.min(x))
        xmax = xave + 1.05 * (np.max(x) - xave)
        if yerr is None:
            yave = 0.5 * (np.max(y) + np.min(y))
            ymin = yave - 1.05 * (yave - np.min(y))
            ymax = yave + 1.05 * (np.max(y) - yave)
            self.binAxes.scatter(x, y, marker='.', c='g')
        else:
            yave = 0.5 * (np.max(y+yerr) + np.min(y-yerr))
            ymin = yave - 1.05 * (yave - np.min(y-yerr))
            ymax = yave + 1.05 * (np.max(y+yerr) - yave)
            self.binAxes.errorbar(x, y, yerr=yerr, fmt='.', color='green')

        self.binAxes.axis([xmin, xmax, ymin, ymax])
        self.binAxes.get_xaxis().get_major_formatter().set_useOffset(False)
        self.binAxes.set_xlabel(xlabel)
        self.binAxes.set_ylabel(ylabel)
        self.binAxes.set_title(title)
        self.binCanvas.draw()
        self.setColorScheme(False)

    def setFocusToCanvas(self):
        """
        Set the focus to the plk Canvas
        """
        self.binCanvas.setFocus()

    def coord2point(self, cx, cy):
        """
        Given data coordinates x and y, obtain the index of the observations
        that is closest to it
        
        @param cx:  x-value of the coordinates
        @param cy:  y-value of the coordinates
        
        @return:    Index of observation
        """
        ind = None

        if self.psr is not None:
            # Get a mask for the plotting points
            msk = self.psr.mask('plot')

            # Get the IDs of the X and Y axis
            #xid, yid = self.xyChoiceWidget.plotids()
            xid, yid = 'MJD', 'post-fit'

            # Retrieve the data
            x, xerr, xlabel = self.psr.data_from_label(xid)
            y, yerr, ylabel = self.psr.data_from_label(yid)

            if np.sum(msk) > 0 and x is not None and y is not None:
                # Obtain the limits
                xmin, xmax, ymin, ymax = self.binAxes.axis()

                dist = ((x[msk]-cx)/(xmax-xmin))**2 + ((y[msk]-cy)/(ymax-ymin))**2
                ind = np.arange(len(x))[msk][np.argmin(dist)]

        return ind


    def keyPressEvent(self, event, **kwargs):
        """
        A key is pressed. Handle all the shortcuts here.

        This function can be called as a callback from the Canvas, or as a
        callback from Qt. So first some parsing must be done
        """

        if hasattr(event.key, '__call__'):
            ukey = event.key()
            modifiers = int(event.modifiers())
            from_canvas = False

            print("WARNING: call-back key-press, canvas location not available")

            xpos, ypos = None, None
        else:
            # Modifiers are noted as: key = 'ctrl+alt+F', or 'alt+control', or
            # 'shift+g'. Do some parsing
            fkey = event.key
            from_canvas = True

            xpos, ypos = event.xdata, event.ydata

            ukey = ord(fkey[-1])
            modifiers = QtCore.Qt.NoModifier
            if 'ctrl' in fkey:
                modifiers += QtCore.Qt.ControlModifier
            if 'shift' in fkey:
                modifiers += QtCore.Qt.ShiftModifier
            if 'alt' in fkey:
                modifiers += QtCore.Qt.ShiftModifier
            if 'meta' in fkey:
                modifiers += QtCore.Qt.MetaModifier

        #if int(e.modifiers()) == (QtCore.Qt.ControlModifier+QtCore.Qt.AltModifier)

        if ukey == QtCore.Qt.Key_Escape:
            if self.parent is None:
                self.close()
            else:
                self.parent.close()
        elif ukey == ord('s'):
            # Set START flag at xpos
            # TODO: propagate back to the IPython shell
            pass
        elif ukey == ord('f'):
            # Set FINISH flag as xpos
            # TODO: propagate back to the IPython shell
            pass
        elif ukey == ord('u'):
            # Unzoom
            # TODO: propagate back to the IPython shell
            pass
        elif ukey == ord('d'):
            # Delete data point
            # TODO: propagate back to the IPython shell
            pass
        elif ukey == ord('x'):
            # Re-do the fit, using post-fit values of the parameters
            pass
        elif ukey == QtCore.Qt.Key_Left:
            # print("Left pressed")
            pass
        else:
            #print("Other key: {0} {1} {2} {3}".format(ukey,
            #    modifiers, ord('M'), QtCore.Qt.ControlModifier))
            pass

        if not from_canvas:
            if self.parent is not None:
                print("Propagating key press")
                self.parent.keyPressEvent(event)

            super(BinaryWidget, self).keyPressEvent(event, **kwargs)

    def canvasClickEvent(self, event):
        """
        When one clicks on the Figure/Canvas, this function is called. The
        coordinates of the click are stored in event.xdata, event.ydata
        """
        #print('Canvas click, you pressed', event.button, event.xdata, event.ydata)
        pass

    def canvasKeyEvent(self, event):
        """
        When one presses a button on the Figure/Canvas, this function is called.
        The coordinates of the click are stored in event.xdata, event.ydata
        """
        # Callback to the binaryWidget
        self.keyPressEvent(event)

