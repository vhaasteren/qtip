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

# For angle conversions
import ephem        # pip install pyephem

import constants
import qtpulsar as qp
import tempfile
from libfitorbit import orbitpulsar, array_to_pardict

import math 
import scipy.optimize as so

# Regular expressions for RA and DEC fields
RAREGEXP = "^(([01]?[0-9]|2[0-4]):([0-5][0-9]):([0-5][0-9])(\.[0-9]+)?)$"
DECREGEXP = "^((-?[0-8]?[0-9]|90):([0-5][0-9]):([0-5][0-9])(\.[0-9]+)?)$"

class BinaryWidget(QtGui.QWidget):
    """
    The binary-widget window.

    @param parent:      Parent window
    """

    def __init__(self, parent=None, parfilename=None, perfilename=None, **kwargs):
        super(BinaryWidget, self).__init__(parent, **kwargs)

        self.initBin()
        #self.openPulsar(parfilename, perfilename)
        self.fillModelPars()
        self.updatePlot()

        self.parent = parent

    def initBin(self):
        """
        Initialize all the Widgets, and add them to the layout
        """
        numpars = 10
        self.parameterCols = 2
        self.parameterRows = int(np.ceil(numpars / self.parameterCols))
        cblength = 6
        inplength = 20

        # Create an empty binary pulsar object (read later)
        self.bpsr = orbitpulsar()
        self.psrLoaded = False
        self.blockModelUpdate = False       # Respond to parameter edits yes/no

        # Create an empty dictionary that will carry the plotting information
        self.plotdict = {}
        self.showplot = None
        self.plotmodel = True

        self.setMinimumSize(650, 550)

        # The layout boxes and corresponding widgets
        self.fullwidgetbox = QtGui.QHBoxLayout()        # whole widget
        self.operationbox = QtGui.QVBoxLayout()         # operation sect. (left)
        self.inoutputbox = QtGui.QVBoxLayout()          # in and output (right)
        self.parameterbox = QtGui.QGridLayout()         # The model parameters

        # The binaryModel Combobox Widget
        self.binaryModelCB = QtGui.QComboBox()
        self.binaryModelCB.addItem('BT')
        self.binaryModelCB.addItem('DD')
        self.binaryModelCB.addItem('T2')
        self.binaryModelCB.addItem('ELL')
        self.binaryModelCB.setCurrentIndex(0)
        #self.binaryModelCB.stateChanged.connect(self.changedBinaryModel)
        self.operationbox.addWidget(self.binaryModelCB)

        # Button for simulating data
        self.simButton = QtGui.QPushButton("Simulate Data")
        self.simButton.clicked.connect(self.simData)
        self.operationbox.addWidget(self.simButton)

        # Button for parfile writing
        self.writeButton = QtGui.QPushButton('Write Par')
        self.writeButton.clicked.connect(self.writePar)
        self.operationbox.addWidget(self.writeButton)

        # Button for fitting the model (least-squares fit)
        self.fitButton = QtGui.QPushButton('Fit Model')
        self.fitButton.clicked.connect(self.fitModel)
        self.operationbox.addWidget(self.fitButton)

        # Checkbox for yes/no plot Model
        self.plotCheckBox = QtGui.QCheckBox('Plot Model')
        self.plotCheckBox.stateChanged.connect(self.changedPlotModel)
        self.operationbox.addWidget(self.plotCheckBox)

        # Button for plotting the periods
        self.periodButton = QtGui.QPushButton('Periods')
        self.periodButton.clicked.connect(self.plotPeriods)
        self.operationbox.addWidget(self.periodButton)

        # Button for the Roughness
        self.roughButton = QtGui.QPushButton('Roughness')
        self.roughButton.clicked.connect(self.plotRough)
        self.operationbox.addWidget(self.roughButton)

        # Finish the operation Widget
        self.operationbox.addStretch(1)
        self.fullwidgetbox.addLayout(self.operationbox)

        # Place the model boxes on a grid
        self.parameterbox = QtGui.QGridLayout()
        self.parameterbox.setSpacing(10)

        # Add all the parameters
        index = 0
        bModel = str(self.binaryModelCB.currentText())
        PARAMS = self.bpsr.bmparams[bModel]
        self.parameterbox_pw = []
        for ii in range(self.parameterRows):
            for jj in range(self.parameterCols):
                if index < numpars:
                    # Add another parameter to the grid
                    offset = jj*(cblength + inplength)

                    checkbox = QtGui.QCheckBox(PARAMS[index], parent=self)
                    checkbox.stateChanged.connect(self.changedParFit)
                    checkbox.setChecked(True)
                    self.parameterbox.addWidget(checkbox, \
                            ii, offset, 1, cblength)

                    # TODO: Figure out how to properly set an edit field
                    lineedit = QtGui.QLineEdit("", parent=self)
                    if PARAMS[index] == 'RA':
                        regexp = QtCore.QRegExp(RAREGEXP)
                        validator = QtGui.QRegExpValidator(regexp)
                    elif PARAMS[index] == 'DEC':
                        regexp = QtCore.QRegExp(DECREGEXP)
                        validator = QtGui.QRegExpValidator(regexp)
                    else:
                        validator = QtGui.QDoubleValidator()
                    lineedit.setValidator(validator)
                    lineedit.textChanged.connect(self.changedPars)
                    self.parameterbox.addWidget(lineedit, \
                            ii, offset+cblength, 1, inplength)

                    # Send the textChanged signal for the color update
                    lineedit.textChanged.emit(lineedit.text())

                    # Save the widgets for later reference
                    self.parameterbox_pw.append(\
                            dict({'checkbox':checkbox, 'lineedit':lineedit}))

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
        self.binAxes1 = self.binFig.add_subplot(211)
        self.binAxes2 = self.binFig.add_subplot(212)

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
        self.inoutputbox.addWidget(self.binCanvas)
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
        self.windCol = (r/255.0, g/255.0, b/255.0)

        if start:
            # Copy of 'white', because of bug in matplotlib that does not allow
            # deep copies of rcParams. Store values of matplotlib.rcParams
            self.orig_rcParams = copy.deepcopy(constants.mpl_rcParams_white)
            for key, value in self.orig_rcParams.iteritems():
                self.orig_rcParams[key] = matplotlib.rcParams[key]

            rcP = copy.deepcopy(constants.mpl_rcParams_white)
            rcP['axes.facecolor'] = self.windCol
            rcP['figure.facecolor'] = self.windCol
            rcP['figure.edgecolor'] = self.windCol
            rcP['savefig.facecolor'] = self.windCol
            rcP['savefig.edgecolor'] = self.windCol

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
        self.binAxes1.clear()
        self.binAxes2.clear()
        self.binCanvas.draw()
        self.setColorScheme(False)

    def setPulsar(self, bpsr):
        """
        Yay, we have got a new pulsar!

        @param bpsr:    New binary pulsar object
        """
        self.bpsr = bpsr
        self.fillModelPars()
        self.psrLoaded = True

        # Erase all the plotting information, and then show the plot
        self.plotdict = {}
        self.showplot = None
        self.setPlotPeriods()
        self.updatePlot()

    def openPulsar(self, parfilename=None, perfilename=None):
        """
        Open a per/par file.

        TODO: This needs to be in the kernel namespace. But for now, keep it in
              the widget
        """
        # TODO: deprecated
        if perfilename is None or parfilename is None:
            # Write temporary files
            tperfilename = tempfile.mktemp()
            tparfilename = tempfile.mktemp()
            tperfile = open(tperfilename, 'w')
            tparfile = open(tparfilename, 'w')
            tperfile.write(constants.J1903PER)
            #tperfile.write(constants.J1756PER)
            tparfile.write(constants.J1903EPH)
            #tparfile.write(constants.J1756EPH)
            tperfile.close()
            tparfile.close()
        else:
            tperfilename = perfilename
            tparfilename = parfilename

        self.bpsr.readParFile(tparfilename)
        self.bpsr.readPerFile(tperfilename, ms=True)

        if perfilename is None or parfilename is None:
            os.remove(tperfilename)
            os.remove(tparfilename)

        self.psrLoaded = True

    def fillModelPars(self):
        """
        When we have read a new pulsar/model, we need to propagate the newly
        read parameters back to the input field. That's what we do here.
        """
        self.blockModelUpdate = True
        for pw in self.parameterbox_pw:
            pid = pw['checkbox'].text()

            if pid in self.bpsr.pars(which='set'):
                if pid == 'RA':
                    pw['lineedit'].setText(str(ephem.hours(self.bpsr[pid].val)))
                elif pid == 'DEC':
                    pw['lineedit'].setText(str(ephem.degrees(self.bpsr[pid].val)))
                else:
                    pw['lineedit'].setText(str(self.bpsr[pid].val))

        self.blockModelUpdate = False

    def getModelPars(self):
        """
        Obtain the binary model parameters from the input fields. If they all
        pass the validator tests, we propagate these values into the parameter
        dictionary
        """
        if self.validateModelPars():
            for pw in self.parameterbox_pw:
                pid = pw['checkbox'].text()

                if pid in self.bpsr.pars(which='set'):
                    if pid == 'RA':
                        self.bpsr[pid].val = \
                            np.float128(ephem.hours(str(pw['lineedit'].text())))
                    elif pid == 'DEC':
                        self.bpsr[pid].val = \
                            np.float128(ephem.degrees(str(pw['lineedit'].text())))
                    else:
                        self.bpsr[pid].val = pw['lineedit'].text()
                else:
                    # Add the parameter, so do some extra stuff?
                    pass

    def validateModelPars(self):
        """
        Returns True when all binary model parameter fields validate
        """
        all_ok = True
        for pw in self.parameterbox_pw:
            # Check the validators first
            validator = pw['lineedit'].validator()
            if validator.validate(pw['lineedit'].text(), 0)[0] != QtGui.QValidator.Acceptable:
                all_ok = False

        return all_ok


    def changedBinaryModel(self):
        """
        Called when we change the state of the Binary Model combobox.
        """
        pass

    def changedPlotModel(self):
        """
        Called when we change the state of the PlotModel checkbox.
        """
        self.plotmodel = bool(self.plotCheckBox.checkState())

        self.updatePlot()

    def changedPars(self, *args, **kwargs):
        """
        Called when we have changed the parameters of the binary model
        """
        sender = self.sender()
        validator = sender.validator()
        state = validator.validate(sender.text(), 0)[0]
        if state == QtGui.QValidator.Acceptable:
            color = '#c4df9b' # green
        elif state == QtGui.QValidator.Intermediate:
            color = '#fff79a' # yellow
        else:
            color = '#f6989d' # red
        sender.setStyleSheet('QLineEdit { background-color: %s }' % color)

        # If all validate, we need to plot the graph again
        if self.validateModelPars():
            if self.psrLoaded and not self.blockModelUpdate:
                self.getModelPars()
                self.createPeriodPlot()
                #self.createRoughnessPlot()
                self.updatePlot()

    def changedParFit(self, *args, **kwargs):
        """
        Called when we haved set/unset a fitting checkbox of the binary model
        """
        for pw in self.parameterbox_pw:
            pid = pw['checkbox'].text()
            self.bpsr[pid].fit = pw['checkbox'].checkState()

    def simData(self):
        """
        Simulate some reasonable data
        """
        nobs = 100
        tmin = np.min(self.bpsr.mjds)
        tmax = np.max(self.bpsr.mjds)
        newmjds = np.linspace(tmin, tmax, nobs) + \
                np.random.randn(nobs) * (tmax-tmin) / (4*nobs)
        self.bpsr.simData(newmjds)

        if self.showplot == 'roughness':
            self.createRoughnessPlot()
        self.createPeriodPlot()
        self.updatePlot()

    def writePar(self):
        """
        Function to write the timing model parameters to file
        """
        fname = QtGui.QFileDialog.getSaveFileName(self, 'Export par-file', \
                '', 'par-file (*.par)')
        fname = self.bpsr.writeParFile(fname)
        print("Par-file written: {0}".format(fname))
        

    def fitModel(self):
        """
        Function to perform the fit of selected parameters to the values
        """
        # Use a mask to keep track of the parameters we fit for
        fitmsk = self.bpsr.parmask(which='fit')

        # TODO: RA and DEC are still in strings. Convert them somewhere good

        # Initialize the parameters
        nfix = np.sum(fitmsk)
        apars = self.bpsr.vals(which='set')
        fpars = self.bpsr.vals(which='fit')
        keys = self.bpsr.pars(which='set')
        pd = array_to_pardict(apars, which='BT')

        for ii, key in enumerate(keys):
            pd[key].val = apars[ii]

        # Create a function for the residuals
        def resids(pars, bpsr, allpars, mask):
            allpars[mask] = pars
            for ii, key in enumerate(keys):
                pd[key].val = allpars[ii]
            res =  bpsr.orbitResiduals(pardict=pd, weight=True, model='BT')
            return res

        # If there are parameters to fit, do a least-squares minimization
        if np.sum(fitmsk) > 0:
            # Perform the least-squares fit
            plsq = so.leastsq(resids, np.float64(fpars),
                    args=(self.bpsr, apars, fitmsk), full_output=True)

            #print("key:", keys)
            #print("par:", plsq[0])
            #print("err:", np.sqrt(np.diag(plsq[1])))

            # Place the new paramerers back in the boxes
            apars[fitmsk] = plsq[0]
            self.bpsr.vals(which='set', newvals=apars)

            self.createPeriodPlot()
            self.updatePlot()

            self.fillModelPars()
        else:
            pass

    def createPeriodPlot(self):
        """
        Create the plotting information for the period plots
        """
        pdp, pdf = {}, {}

        # The period plot
        xs = np.linspace(np.min(self.bpsr.mjds), np.max(self.bpsr.mjds), 2000)
        ys = self.bpsr.orbitModel(mjds=xs)
        pdp['plot'] = (xs, ys)
        if self.bpsr.periodserrs is None:
            pdp['scatter'] = (self.bpsr.mjds, self.bpsr.periods)
        else:
            pdp['errorbar'] = (self.bpsr.mjds, self.bpsr.periods, \
                    self.bpsr.periodserrs)
        if len(self.bpsr.mjds) > 0:
            dx = 0.05 * (max(self.bpsr.mjds) - min(self.bpsr.mjds))
            dy = 0.05 * (max(self.bpsr.periods) - min(self.bpsr.periods))
            xlims = (min(self.bpsr.mjds)-dx, max(self.bpsr.mjds)+dx)
            ylims = (min(self.bpsr.periods)-dy, max(self.bpsr.periods)+dy)
        else:
            xlims = (0.0, 1.0)
            ylims = (0.0, 1.0)
        pdp['xlim'] = xlims
        pdp['ylim'] = ylims
        pdp['xlabel'] = 'MJD'
        pdp['ylabel'] = 'Pulse period (ms)'

        # The orbit per phase plot
        per = np.float64(self.bpsr['PB'].val)
        phi = np.fmod(np.float64(self.bpsr['T0'].val), per)
        phase = np.fmod(self.bpsr.mjds-phi, per) / per
        xphase = np.fmod(xs-phi, per) / per
        xinds = np.argsort(xphase)[::-1]
        pdf['plot'] = (xphase[xinds], ys[xinds])
        if self.bpsr.periodserrs is None:
            pdf['scatter'] = (phase, self.bpsr.periods)
        else:
            pdf['errorbar'] = (phase, self.bpsr.periods, \
                    self.bpsr.periodserrs)
        pdf['xlim'] = (0.0, 1.0)
        pdf['ylim'] = ylims
        pdf['xlabel'] = 'Phase'
        pdf['ylabel'] = 'Pulse period (ms)'

        self.plotdict['period'] = pdp
        self.plotdict['phase'] = pdf

    def createRoughnessPlot(self, spacing='cubic', useCurrent=True):
        """
        Create the plotting information for the roughness plots

        @param spacing:     'cubic'/'log' spacing for Pb
        @param useCurrent:  Use current or estimated values for parameters
        """
        pdr, pdf = {}, {}
        pdeo, pdi = {}, {}

        # Estimate the roughness
        pb, rg = self.bpsr.roughnessPlot(pbmin=1.0, pbmax=20.0)

        # Roughness
        pdr['scatter'] = (np.log10(pb), np.log10(rg))
        pdr['xlabel'] = 'log10(Period)'
        pdr['ylabel'] = 'log10(Roughness)'
        pdr['xlim'] = (np.log10(min(pb)), np.log10(max(pb)))
        pdr['ylim'] = (np.log10(min(rg)), np.log10(max(rg)))

        # Phase roughness at best frequency
        minind = np.argmin(rg)
        pbm = pb[minind]
        phi = np.fmod(np.float64(self.bpsr['T0'].val), pbm)
        phase = np.fmod(self.bpsr.mjds-phi, pbm) / pbm
        inds = np.argsort(phase)
        pdf['scatter'] = (phase[inds], self.bpsr.periods[inds])
        pdf['annotate'] = "Pb = {0:.2f}".format(pbm)
        pdf['xlabel'] = 'Phase'
        pdf['ylabel'] = 'Pulse period (ms)'
        pdf['xlim'] = (0.0, 1.0)
        pdf['ylim'] = (min(self.bpsr.periods), max(self.bpsr.periods))

        # Even/odd calculations
        if useCurrent:
            # Use the provided values to make the Even/Odd plots
            Pe, Po = self.bpsr.Peo(np.float64(self.bpsr['PB'].val), \
                    T0=np.float64(self.bpsr['T0'].val), kind='linear')
            Pei, Poi, phasei, fi = \
                    self.bpsr.Peo_interp(np.float64(self.bpsr['PB'].val), \
                    T0=np.float64(self.bpsr['T0'].val), kind='linear')
        else:
            pass
        pdeo['plotcur'] = Pe, Po
        pdeo['plotint'] = Pei, Poi
        #pdeo['plotest'] = self.bpsr.Peo(PBest, T0est)
        #pdeo['xlim'] = np.min(Pei), np.max(Pei)
        #pdeo['ylim'] = np.min(Poi), np.max(Poi)
        pdeo['xlim'] = 2*np.min(Pe)-np.max(Pe), 2*np.max(Pe)-np.min(Pe)
        pdeo['ylim'] = 2*np.min(Po)-np.max(Po), 2*np.max(Po)-np.min(Po)
        #pdeo['xlim'] = np.min(Pe), np.max(Pe)
        #pdeo['ylim'] = np.min(Po), np.max(Po)
        pdeo['xlabel'] = 'P_e'
        pdeo['ylabel'] = 'P_o'

        per = np.float64(self.bpsr['PB'].val)
        phi = np.fmod(np.float64(self.bpsr['T0'].val), per)
        phase = np.fmod(self.bpsr.mjds-phi, per) / per
        pdi['scatter'] = (phase, self.bpsr.periods)
        pdi['plot'] = (phasei, fi)
        pdi['xlabel'] = 'Phase'
        pdi['ylabel'] = 'Pulse period (ms)'
        pdi['xlim'] = (0.0, 1.0)
        pdi['ylim'] = (min(self.bpsr.periods), max(self.bpsr.periods))

        self.plotdict['roughness'] = pdr
        self.plotdict['roughphase'] = pdf
        self.plotdict['evenodd'] = pdeo
        self.plotdict['phaseint'] = pdi

    def setPlotPeriods(self):
        """
        Set the plot information to plot periods, and create the plot if
        necessary
        """
        if not ('period' in self.plotdict and 'phase' in self.plotdict):
            self.createPeriodPlot()

        self.showplot = 'periodphase'

    def plotPeriods(self):
        """
        Show the period plots, create the plot if necessary, and update the plot
        """
        self.setPlotPeriods()
        self.updatePlot()

    def setPlotRough(self):
        """
        Set the plot information to plot roughness, and create the plot if
        necessary
        """
        #if not ('roughness' in self.plotdict and 'roughphase' in self.plotdict):
        self.createRoughnessPlot()

        self.showplot = 'roughness'

    def plotRough(self):
        """
        Show the roughness plots. Create the plot if necessary
        """
        self.setPlotRough()
        self.updatePlot()

    def updatePlot(self):
        """
        Update the active plot
        """
        if self.showplot is not None:
            # Do not plot anything
            self.setColorScheme(True)
            self.binFig.clf()

            if self.showplot == 'periodphase':
                self.binAxes1 = self.binFig.add_subplot(211)
                self.binAxes2 = self.binFig.add_subplot(212)
                self.binAxes1.grid(True)
                self.binAxes2.grid(True)

                pd = self.plotdict['period']
                self.binAxes1.get_yaxis().get_major_formatter().set_useOffset(False)
                if 'scatter' in pd:
                    self.binAxes1.scatter(*pd['scatter'], c='darkred', marker='.', s=50)
                else:
                    self.binAxes1.errorbar(pd['errorbar'][0], pd['errorbar'][1], \
                            yerr=pd['errorbar'][2], c='blue', fmt='.')
                self.binAxes1.set_xlabel(pd['xlabel'])
                self.binAxes1.set_ylabel(pd['ylabel'])
                self.binAxes1.set_xlim(*pd['xlim'])
                self.binAxes1.set_ylim(*pd['ylim'])
                self.binAxes1.yaxis.labelpad = -1
                if self.plotmodel:
                    self.binAxes1.plot(*pd['plot'], c='r', linestyle='-')

                pd = self.plotdict['phase']
                self.binAxes2.get_yaxis().get_major_formatter().set_useOffset(False)
                if 'scatter' in pd:
                    self.binAxes2.scatter(*pd['scatter'], c='darkred', marker='.', s=50)
                else:
                    self.binAxes2.errorbar(pd['errorbar'][0], pd['errorbar'][1], \
                            yerr=pd['errorbar'][2], c='blue', fmt='.')
                self.binAxes2.set_xlabel(pd['xlabel'])
                self.binAxes2.set_ylabel(pd['ylabel'])
                self.binAxes2.set_xlim(*pd['xlim'])
                self.binAxes2.set_ylim(*pd['ylim'])
                self.binAxes2.yaxis.labelpad = -1
                if self.plotmodel:
                    self.binAxes2.plot(*pd['plot'], c='r', linestyle='-')
            elif self.showplot == 'roughness':
                self.binAxes1 = self.binFig.add_subplot(221)
                self.binAxes2 = self.binFig.add_subplot(222)
                self.binAxes3 = self.binFig.add_subplot(223)
                self.binAxes4 = self.binFig.add_subplot(224)
                #self.binAxes1.grid(True)
                #self.binAxes2.grid(True)

                pd = self.plotdict['roughness']
                #self.binAxes1.get_yaxis().get_major_formatter().set_useOffset(False)
                self.binAxes1.scatter(*pd['scatter'], c='darkred', marker='.', s=10)
                self.binAxes1.set_xlabel(pd['xlabel'])
                self.binAxes1.set_ylabel(pd['ylabel'])
                self.binAxes1.set_xlim(*pd['xlim'])
                self.binAxes1.set_ylim(*pd['ylim'])
                self.binAxes1.yaxis.labelpad = -1

                pd = self.plotdict['roughphase']
                #self.binAxes2.get_yaxis().get_major_formatter().set_useOffset(False)
                self.binAxes2.scatter(*pd['scatter'], c='darkred', marker='.', s=50)
                self.binAxes2.set_xlabel(pd['xlabel'])
                self.binAxes2.set_ylabel(pd['ylabel'])
                self.binAxes2.set_xlim(*pd['xlim'])
                self.binAxes2.set_ylim(*pd['ylim'])
                self.binAxes2.yaxis.labelpad = -1
                self.binAxes2.annotate(pd['annotate'], xy=(0.04, 0.81), \
                        xycoords='axes fraction', bbox=dict(boxstyle="round", \
                        fc=self.windCol))

                pd = self.plotdict['evenodd']
                self.binAxes3.plot(*pd['plotint'], c='g', linestyle='-')
                self.binAxes3.scatter(*pd['plotcur'], c='darkred', marker='.', s=50)
                #self.binAxes3.scatter(*pd['plotest'], c='green', marker='.', s=50)
                self.binAxes3.set_xlabel(pd['xlabel'])
                self.binAxes3.set_ylabel(pd['ylabel'])
                self.binAxes3.set_xlim(*pd['xlim'])
                self.binAxes3.set_ylim(*pd['ylim'])

                pd = self.plotdict['phaseint']
                self.binAxes4.scatter(*pd['scatter'], c='darkred', marker='.', s=50)
                self.binAxes4.plot(*pd['plot'], c='g', linestyle='-')
                self.binAxes4.set_xlabel(pd['xlabel'])
                self.binAxes4.set_ylabel(pd['ylabel'])
                self.binAxes4.set_xlim(*pd['xlim'])
                self.binAxes4.set_ylim(*pd['ylim'])
                self.binAxes4.yaxis.labelpad = -1

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

        if self.bpsr is not None:
            # Get a mask for the plotting points
            msk = self.bpsr.mask('plot')

            # Get the IDs of the X and Y axis
            #xid, yid = self.xyChoiceWidget.plotids()
            xid, yid = 'MJD', 'post-fit'

            # Retrieve the data
            x, xerr, xlabel = self.bpsr.data_from_label(xid)
            y, yerr, ylabel = self.bpsr.data_from_label(yid)

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

