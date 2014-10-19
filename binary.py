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



class BinaryWidget(QtGui.QWidget):
    """
    The binary-widget window.

    @param parent:      Parent window
    """

    def __init__(self, parent=None, **kwargs):
        super(BinaryWidget, self).__init__(parent, **kwargs)

        self.initBin()
        self.initBinLayout()
        self.showVisibleWidgets()

        self.psr = None
        self.parent = parent

    def initBin(self):
        self.setMinimumSize(650, 550)

        self.binarybox = QtGui.QVBoxLayout()                       # binarybox contains the whole plk widget
        #self.xyplotbox = QtGui.QHBoxLayout()                    # plkbox contains the whole plk widget
        #self.fitboxesWidget = BinFitboxesWidget(parent=self)    # Contains all the checkboxes
        #self.actionsWidget = BinActionsWidget(parent=self)

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

        # Create the navigation toolbar, tied to the canvas
        #
        #self.mpl_toolbar = NavigationToolbar(self.canvas, self.main_frame)

        # Draw an empty graph
        self.drawSomething()

        # Create the XY choice widget
        #self.xyChoiceWidget = BinXYPlotWidget(parent=self)

        # At startup, all the widgets are visible
        self.xyChoiceVisible = True
        self.fitboxVisible = True
        self.actionsVisible = True
        self.layoutMode = 1         # (0 = none, 1 = all, 2 = only fitboxes, 3 = fit & action)

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
        #self.binAxes.clear()
        #self.binAxes.grid(True)
        #self.binAxes.set_xlabel('MJD')
        #self.binAxes.set_ylabel('Residual ($\mu$s)')
        #self.binCanvas.draw()
        self.setColorScheme(False)

    def setPulsar(self, psr):
        """
        We've got a new pulsar!
        """
        self.psr = psr

        # Update the fitting checkboxes
        #self.fitboxesWidget.setPulsar(psr)
        #self.xyChoiceWidget.setPulsar(psr, self.updatePlot)
        #self.actionsWidget.setPulsar(psr, self.updatePlot, self.reFit)

        # Draw the residuals
        #self.xyChoiceWidget.updateChoice()
        self.show()

    def reFit(self):
        """
        We need to re-do the fit for this pulsar
        """
        if not self.psr is None:
            self.psr.fit()
            self.updatePlot()

    def newFitParameters(self):
        """
        This function is called when we have new fitparameters

        TODO: callback not used right now
        """
        pass

    def initBinLayout(self):
        """
        Initialise the basic layout of this plk emulator emulator
        """
        # Initialise the plk box
        #self.plkbox.addWidget(self.fitboxesWidget)

        #self.xyplotbox.addWidget(self.xyChoiceWidget)
        #self.xyplotbox.addWidget(self.binCanvas)

        #self.plkbox.addLayout(self.xyplotbox)

        #self.plkbox.addWidget(self.actionsWidget)
        self.setLayout(self.binarybox)

    def showVisibleWidgets(self):
        """
        Show the correct widgets in the plk Window
        """
        #self.xyChoiceWidget.setVisible(self.xyChoiceVisible)
        #self.fitboxesWidget.setVisible(self.fitboxVisible)
        #self.actionsWidget.setVisible(self.actionsVisible)


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
        elif (ukey == ord('M') or ukey == ord('m')) and \
                modifiers == QtCore.Qt.ControlModifier:
            # Change the window
            self.layoutMode = (1+self.layoutMode)%4
            if self.layoutMode == 0:
                self.xyChoiceVisible = False
                self.fitboxVisible = False
                self.actionsVisible = False
            elif self.layoutMode == 1:
                self.xyChoiceVisible = True
                self.fitboxVisible = True
                self.actionsVisible = True
            elif self.layoutMode == 2:
                self.xyChoiceVisible = False
                self.fitboxVisible = True
                self.actionsVisible = True
            elif self.layoutMode == 3:
                self.xyChoiceVisible = False
                self.fitboxVisible = True
                self.actionsVisible = False
            self.showVisibleWidgets()
        elif ukey == ord('s'):
            # Set START flag at xpos
            # TODO: propagate back to the IPython shell
            self.psr['START'].set = True
            self.psr['START'].fit = True
            self.psr['START'].val = xpos
            self.updatePlot()
        elif ukey == ord('f'):
            # Set FINISH flag as xpos
            # TODO: propagate back to the IPython shell
            self.psr['FINISH'].set = True
            self.psr['FINISH'].fit = True
            self.psr['FINISH'].val = xpos
            self.updatePlot()
        elif ukey == ord('u'):
            # Unzoom
            # TODO: propagate back to the IPython shell
            self.psr['START'].set = True
            self.psr['START'].fit = False
            self.psr['START'].val = np.min(self.psr.toas)
            self.psr['FINISH'].set = True
            self.psr['FINISH'].fit = False
            self.psr['FINISH'].val = np.max(self.psr.toas)
            self.updatePlot()
        elif ukey == ord('d'):
            # Delete data point
            # TODO: propagate back to the IPython shell
            # TODO: Fix libstempo!
            ind = self.coord2point(xpos, ypos)
            #print("Deleted:", self.psr._psr.deleted)
            # TODO: fix this hack properly in libstempo
            tempdel = self.psr.deleted
            tempdel[ind] = True
            self.psr.deleted = tempdel
            self.updatePlot()
            #print("Index deleted = ", ind)
            #print("Deleted:", self.psr.deleted[ind])
        elif ukey == ord('x'):
            # Re-do the fit, using post-fit values of the parameters
            self.reFit()
        elif ukey == QtCore.Qt.Key_Left:
            # print("Left pressed")
            pass
        else:
            #print("Other key: {0} {1} {2} {3}".format(ukey,
            #    modifiers, ord('M'), QtCore.Qt.ControlModifier))
            pass

        #print("BinaryWidget: key press: ", ukey, xpos, ypos)

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

