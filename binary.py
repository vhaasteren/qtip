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

# For angle conversions
import ephem        # pip install pyephem

import constants
import qtpulsar as qp
import tempfile
from libfitorbit import orbitpulsar

import math 
from scipy.optimize import leastsq

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
        self.openPulsar(parfilename, perfilename)
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
        self.psr = orbitpulsar()
        self.psrLoaded = False

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
        bModel = str(self.binaryModelCB.currentText())
        PARAMS = self.psr.bmparams[bModel]
        self.parameterbox_pw = []
        for ii in range(self.parameterRows):
            for jj in range(self.parameterCols):
                if index < numpars:
                    # Add another parameter to the grid
                    offset = jj*(cblength + inplength)

                    checkbox = QtGui.QCheckBox(PARAMS[index], parent=self)
                    checkbox.stateChanged.connect(self.changedParFit)
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
        self.binAxes.set_ylabel('Pulse period (ms)')
        self.binAxes.yaxis.labelpad = -1
        self.binCanvas.draw()
        self.setColorScheme(False)

    def openPulsar(self, parfilename=None, perfilename=None):
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
            #tperfile.write(constants.J1903PER)
            tperfile.write(constants.J1756PER)
            #tparfile.write(constants.J1903EPH)
            tparfile.write(constants.J1756EPH)
            tperfile.close()
            tparfile.close()
        else:
            tperfilename = perfilename
            tparfilename = parfilename

        self.psr.readParFile(tparfilename)
        self.psr.readPerFile(tperfilename)

        if perfilename is None or parfilename is None:
            os.remove(tperfilename)
            os.remove(tparfilename)

        self.psrLoaded = True

    def fillModelPars(self):
        """
        When we have read a new pulsar/model, we need to propagate the newly
        read parameters back to the input field. That's what we do here.
        """
        for pw in self.parameterbox_pw:
            pid = pw['checkbox'].text()

            if pid in self.psr.pars(which='set'):
                if pid == 'RA':
                    pw['lineedit'].setText(str(ephem.hours(self.psr[pid].val)))
                elif pid == 'DEC':
                    pw['lineedit'].setText(str(ephem.degrees(self.psr[pid].val)))
                else:
                    pw['lineedit'].setText(str(self.psr[pid].val))

    def getModelPars(self):
        """
        Obtain the binary model parameters from the input fields. If they all
        pass the validator tests, we propagate these values into the parameter
        dictionary
        """
        all_ok = True
        for pw in self.parameterbox_pw:
            # Check the validators first
            validator = pw['lineedit'].validator()
            if validator.validate(pw['lineedit'].text(), 0)[0] != QtGui.QValidator.Acceptable:
                all_ok = False

        if all_ok:
            for pw in self.parameterbox_pw:
                pid = pw['checkbox'].text()

                if pid in self.psr.pars(which='set'):
                    if pid == 'RA':
                        self.psr[pid].val = \
                            np.float128(ephem.hours(str(pw['lineedit'].text())))
                    elif pid == 'DEC':
                        self.psr[pid].val = \
                            np.float128(ephem.degrees(str(pw['lineedit'].text())))
                    else:
                        self.psr[pid].val = pw['lineedit'].text()
                else:
                    # Add the parameter, so do some extra stuff?
                    pass
        

    def plotModel(self, widget=None):
        """
        Plot the best-fit binary model
        """
        xs = np.linspace(min(self.psr.mjds), max(self.psr.mjds), 2000)
        ys = self.psr.orbitModel(mjds=xs)
        
        # Redraw plot
        self.setColorScheme(True)
        self.binAxes.clear()
        self.binAxes.grid(True)
        self.binAxes.get_yaxis().get_major_formatter().set_useOffset(False)
        self.binAxes.set_xlabel('MJD')
        self.binAxes.plot(xs, ys, 'r-')
        self.binAxes.scatter(self.psr.mjds, self.psr.periods, \
                c='darkred', marker='.', s=50)
        self.binAxes.set_ylabel('Pulse period (ms)')
        self.binAxes.yaxis.labelpad = -1
        self.binCanvas.draw()
        self.setColorScheme(False)


    def changedBinaryModel(self):
        """
        Called when we change the state of the Binary Model combobox.
        """
        pass

    def changedPlotModel(self):
        """
        Called when we change the state of the PlotModel checkbox.
        """
        if bool(self.plotCheckBox.checkState()):
            self.getModelPars()
            self.plotModel()
        else:
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

    def changedParFit(self, *args, **kwargs):
        """
        Called when we haved set/unset a fitting checkbox of the binary model
        """
        for pw in self.parameterbox_pw:
            pid = pw['checkbox'].text()
            self.psr[pid].fit = pw['checkbox'].checkState()

    def fitModel(self):
        """
        Function to perform the fit of selected parameters to the values
        """
        # Use a mask to keep track of the parameters we fit for
        fitmsk = self.psr.parmask(which='fit')

        # TODO: RA and DEC are still in strings. Convert them somewhere good

        # Initialize the parameters
        nfix = np.sum(fitmsk)
        apars = self.psr.vals(which='set')
        fpars = self.psr.vals(which='fit')

        # Create a function for the residuals
        def resids(pars, psr, allpars, mask):
            allpars[mask] = pars
            return psr.orbitResiduals(parameters=allpars)

        # If there are parameters to fit, do a least-squares minimization
        if np.sum(fitmsk) > 0:
            # Perform the least-squares fit
            plsq = leastsq(resids, np.float64(fpars), args=(self.psr, apars, fitmsk))

            # Place the new paramerers back in the boxes
            apars[fitmsk] = plsq[0]
            self.psr.vals(which='set', newvals=apars)
            self.fillModelPars()
            self.plotModel()
        else:
            pass

    def updatePlot(self):
        """
        Update the plot/figure
        """
        self.setColorScheme(True)
        self.binAxes.clear()
        self.binAxes.grid(True)

        self.binAxes.get_yaxis().get_major_formatter().set_useOffset(False)
        self.binAxes.scatter(self.psr.mjds, self.psr.periods, \
                c='darkred', marker='.', s=50)

        self.binAxes.set_xlabel('MJD')
        self.binAxes.set_ylabel('Pulse period (ms)')
        self.binAxes.yaxis.labelpad = -1
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

