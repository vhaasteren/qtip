#!/usr/bin/python
# -*- coding: utf-8 -*-
# vim: tabstop=4:softtabstop=4:shiftwidth=4:expandtab

"""
Plk: Qt interactive emulator of the tempo2 plk emulator


Help from tempo2 plk:

Fitting and Calculating Options
===============================
b          Bin TOAs within certain time bin
c          Change fitting parameters
d (or right mouse) delete point
ctrl-d     delete highlighted points
e          multiply all TOA errors by given amount
F          run FITWAVES
ctrl-f     remove FITWAVES curve from residuals
i (or left mouse) identify point
M          toggle removing mean from the residuals
ctrl-n     Add white noise to site-arrival-times
p          Change model parameter values
ctrl-p     Toggle plotting versus pulse phase
r          Reset (reload .par and .tim file)
ctrl-r     Select regions in MJDs and write to file
w          toggle fitting using weights
x          redo fit using post-fit parameters
+          add positive phase jump
-          add negative phase jump
BACKSPACE  remove all phase jumps
ctrl-=     add period to residuals above cursor
/          re-read .par file

Plot Selection
==============
D (or middle mouse) view profile
s          start of zoom section
f          finish of zoom section
Ctrl-u     Overplot Shapiro delay
u          unzoom
v          view profiles for highlighted points
V          define the user parameter
Ctrl-v     for pre-fit plotting, decompose the timing model fits
           (i.e. overplot the fitted curves - only for prefit plots
ctrl-X     select x-axis specifically
y          Rescale y-axis only
Y          set y-scale exactly
ctrl-Y     select y-axis specifically
z          Zoom using mouse
<          in zoom mode include previous observation
>          in zoom mode include next observation
1          plot pre-fit  residuals vs date
2          plot post-fit residuals vs date
3          plot pre-fit  residuals vs orbital phase
4          plot post-fit residuals vs orbital phase
5          plot pre-fit  residuals serially
6          plot post-fit residuals serially
7          plot pre-fit  residuals vs day of year
8          plot post-fit residuals vs day of year
9          plot pre-fit  residuals vs frequency
a          plot post-fit residuals vs frequency
!          plot pre-fit  residuals vs TOA error
@          plot post-fit residuals vs TOA error
#          plot pre-fit  residuals vs user values
$          plot post-fit residuals vs user values
%          plot pre-fit  residuals vs year
^          plot post-fit residuals vs year
&          plot pre-fit residuals vs elevation
*          plot post-fit residuals vs elevation
(          plot pre-fit residuals vs rounded MJD
)          plot post-fit residuals vs rounded MJD

Options for selecting x and y axes individually
Ctrl-X n   set x-axis
Ctrl-Y n   set y-axis
where n = 

1         plot pre-fit residuals
2         plot post-fit residuals
3         plot centred MJD
4         plot orbital phase
5         plot TOA number
6         plot day of year
7         plot frequency
8         plot TOA error
9         plot user value
0         plot year
-         plot elevation

Display Options
===============
B          place periodic marks on the x-scale
ctrl-c     Toggle between period epoch and centre for the reference epoch
E          toggle plotting error bars
g          change graphics device
G          change gridding on graphics device
ctrl-e     highlight points more than 3 sigma from the mean
H          highlight points with specific flag using symbols
ctrl-i     highlight points with specific flag using colours
I          indicate individual observations
j          draw line between points 
J          toggle plotting points
L          add label to plot
ctrl-l     add line to plot
ctrl-m     toggle menu bar
N          highlight point with a given filename
o          obtain/highlight all points currently in plot
ctrl-T     set text size
U          unhighlight selected points
[          toggle plotting x-axis on log scale
]          toggle plotting y-axis on log scale

Output Options
==============
Ctrl-J     output listing of residuals in Jodrell format
Ctrl-O     output listing of residuals in simple format
l          list all data points in zoomed region
m          measure distance between two points
P          write new .par file
Ctrl-w     over-write input .par file
S          save a new .tim file
Ctrl-S     overwrite input.tim file
t          Toggle displaying statistics for zoomed region
Ctrl-z     Listing of all highlighted points

Various Options
===============
C          run unix command with filenames for highlighted observations
h          this help file
q          quit



"""

from __future__ import print_function
from __future__ import division
import os, sys

# Importing all the stuff for the IPython console widget
#from PyQt4 import QtGui, QtCore
from qtconsole.qt import QtCore, QtGui

# Importing all the stuff for the matplotlib widget
import matplotlib
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
#from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure

# Numpy etc.
import numpy as np
import copy

import constants
import qtpulsar as qp


# Design philosophy:
# - The pulsar timing engine is dealt with through derivatives of the abstract
# base class 'BasePulsar'. The object is called psr. Interface is close to that
# of libstempo, but it has some extra features related to plotting parameters.
# PlkWidget has psr as a member, but all the child widgets should not (they do
# know about psr at the moment).
# The plotting parameters and all that are obtained through the psr object. No
# calculations whatsoever are supposed to be done in PlkWidget, or it's child
# widget. They need to know as little as possible, so that they are reusable in
# other GUI types.
# Drawing is done through PlkWidget. There is a callback function 'updatePlot'
# that all child widgets are allowed to call, but they should not get access to
# any further data.
# TODO: remove dependence on psr object in child widgets

class PlkActionsWidget(QtGui.QWidget):
    """
    A widget that shows some action items, like re-fit, write par, write tim,
    etc.  These items are shown as buttons
    """

    def __init__(self, parent=None, **kwargs):
        super(PlkActionsWidget, self).__init__(parent, **kwargs)

        self.parent = parent
        self.updatePlot = None
        self.reFit_callback = None

        self.hbox = QtGui.QHBoxLayout()     # One horizontal layout

        self.setPlkActionsWidget()

    def setPlkActionsWidget(self):
        button = QtGui.QPushButton('Re-fit')
        button.clicked.connect(self.reFit)
        self.hbox.addWidget(button)

        button = QtGui.QPushButton('Clear')
        button.clicked.connect(self.clearAll)
        self.hbox.addWidget(button)

        button = QtGui.QPushButton('Write par')
        button.clicked.connect(self.writePar)
        self.hbox.addWidget(button)

        button = QtGui.QPushButton('Write tim')
        button.clicked.connect(self.writeTim)
        self.hbox.addWidget(button)

        button = QtGui.QPushButton('Save fig')
        button.clicked.connect(self.saveFig)
        self.hbox.addWidget(button)

        self.hbox.addStretch(1)

        self.setLayout(self.hbox)

    def setCallbacks(self, updatePlot, reFit):
        """Callback functions"""

        self.updatePlot = updatePlot
        self.reFit_callback = reFit

    def reFit(self):
        if self.reFit_callback is not None:
            self.reFit_callback()

    def writePar(self):
        print("Write Par clicked")

    def writeTim(self):
        print("Write Tim clicked")

    def clearAll(self):
        print("Clear clicked")

    def saveFig(self):
        print("Save fig clicked")


class PlkFitboxesWidget(QtGui.QWidget):
    """A widget that allows one to select which parameters to fit for"""

    def __init__(self, parent=None, **kwargs):
        super(PlkFitboxesWidget, self).__init__(parent, **kwargs)

        self.parent = parent
        self.boxChecked = None

        # The checkboxes are ordered on a grid
        self.hbox = QtGui.QHBoxLayout()     # One horizontal layout
        self.vboxes = []                    # Several vertical layouts (9 per line)
        self.fitboxPerLine = 8

        self.setPlkFitboxesLayout()


    def setPlkFitboxesLayout(self):
        """Set the layout of this widget"""

        # Initialise the layout of the fit-box Widget
        # Initially there are no fitboxes, so just add the hbox
        self.setLayout(self.hbox)


    def setCallbacks(self, boxChecked, setpars, fitpars, nofitbox):
        """Set the callback functions

        :param boxChecked:      Callback function, when box is checked
        :param setpars:         psr.parameters(which='set')
        :param fitpars:         psr.parameters(which='fit')
        :param nofitbox:        Which parameters not to have a box for
        """

        self.boxChecked = boxChecked
        self.deleteFitCheckBoxes()
        self.addFitCheckBoxes(setpars, fitpars, nofitbox)

    def addFitCheckBoxes(self, setpars, fitpars, nofitbox):
        """Add the fitting checkboxes at the top of the plk Window

        :param setpars:     The parameters that are 'set' (in the model)
        :param fitpars:     The parameters that are currently being fitted for
        :param nofitbox:    The parameters we should skip
        """

        # Delete the fitboxes if there were still some left
        if not len(self.vboxes) == 0:
            self.deleteFitCheckBoxes()

        # First add all the vbox layouts
        for ii in range(min(self.fitboxPerLine, len(setpars))):
            self.vboxes.append(QtGui.QVBoxLayout())
            self.hbox.addLayout(self.vboxes[-1])

        # Then add the checkbox widgets to the vboxes
        index = 0
        for pp, par in enumerate(setpars):
            if not par in nofitbox:
                vboxind = index % self.fitboxPerLine

                cb = QtGui.QCheckBox(par, self)
                cb.stateChanged.connect(self.changedFitCheckBox)

                # Set checked/unchecked
                cb.setChecked(par in fitpars)

                self.vboxes[vboxind].addWidget(cb)
                index += 1

        for vv, vbox in enumerate(self.vboxes):
            vbox.addStretch(1)


    def deleteFitCheckBoxes(self):
        """Delete all the checkboxes from the Widget. (for new pulsar)"""

        for fcbox in self.vboxes:
            while fcbox.count():
                item = fcbox.takeAt(0)
                if isinstance(item, QtGui.QWidgetItem):
                    item.widget().deleteLater()
                elif isinstance(item, QtGui.QSpacerItem):
                    fcbox.removeItem(item)
                else:
                    fcbox.clearLayout(item.layout())
                    fcbox.removeItem(item)


        for fcbox in self.vboxes:
            self.hbox.removeItem(fcbox)

        self.vboxes = []

    def changedFitCheckBox(self):
        """This is the signal handler when a checkbox is changed. The changed checkbox
        value will be propagated back to the psr object."""

        # Check who sent the signal
        sender = self.sender()
        parchanged = sender.text()

        # Whatevs, we can just as well re-scan all the CheckButtons, and re-do
        # the fit
        for fcbox in self.vboxes:
            items = (fcbox.itemAt(i) for i in range(fcbox.count())) 
            for w in items:
                if isinstance(w, QtGui.QWidgetItem) and \
                        isinstance(w.widget(), QtGui.QCheckBox) and \
                        parchanged == w.widget().text():
                    self.boxChecked(parchanged, bool(w.widget().checkState()))
                    print("{0} set to {1}".format(parchanged, bool(w.widget().checkState())))



class PlkXYPlotWidget(QtGui.QWidget):
    """
    A widget that allows one to choose which quantities to plot against each other
    """

    def __init__(self, parent=None, **kwargs):
        super(PlkXYPlotWidget, self).__init__(parent, **kwargs)

        self.parent = parent

        # We are going to use a grid layout:
        self.grid = QtGui.QGridLayout()
        self.grid.setSpacing(10)

        self.xButtonGroup = QtGui.QButtonGroup(self)
        self.xButtonGroup.buttonClicked[int].connect(self.updateXPlot)
        self.yButtonGroup = QtGui.QButtonGroup(self)
        self.yButtonGroup.buttonClicked[int].connect(self.updateYPlot)

        self.xSelected = 0
        self.ySelected = 0

        # TODO: implement this:
        # Connect the 'buttonClicked' signal 'self.setLabel'
        # There are two overloads for 'buttonClicked' signal: QAbstractButton (button itself) or int (id)
        # Specific overload for the signal is selected via [QtGui.QAbstractButton]
        # Clicking any button in the QButtonGroup will send this signal with the button
        # self.buttonGroup.buttonClicked[QtGui.QAbstractButton].connect(self.setLabel)
        # def setLabel(self, button):

        # Use an empty base pulsar to obtain the labels
        psr = qp.BasePulsar()
        self.xychoices = psr.plot_labels
    
        self.setPlkXYPlotLayout()

    def setPlkXYPlotLayout(self):
        labellength = 3

        label = QtGui.QLabel(self)
        label.setText("")
        self.grid.addWidget(label, 0, 0, 1, labellength)
        label = QtGui.QLabel(self)
        label.setText("X")
        self.grid.addWidget(label, 0, 0+labellength, 1, 1)
        label = QtGui.QLabel(self)
        label.setText("Y")
        self.grid.addWidget(label, 0, 1+labellength, 1, 1)

        # Add all the xychoices
        for ii, choice in enumerate(self.xychoices):
            # The label of the choice
            label = QtGui.QLabel(self)
            label.setText(choice)
            self.grid.addWidget(label, 1+ii, 0, 1, labellength)

            # The X and Y radio buttons
            radio = QtGui.QRadioButton("")
            self.grid.addWidget(radio, 1+ii, labellength, 1, 1)
            if choice.lower() == 'mjd':
                radio.setChecked(True)
                self.xSelected = ii
            self.xButtonGroup.addButton(radio)
            self.xButtonGroup.setId(radio, ii)

            radio = QtGui.QRadioButton("")
            self.grid.addWidget(radio, 1+ii, 1+labellength, 1, 1)
            if choice.lower() == 'post-fit':
                radio.setChecked(True)
                self.ySelected = ii
            self.yButtonGroup.addButton(radio)
            self.yButtonGroup.setId(radio, ii)

        self.setLayout(self.grid)

    def setCallbacks(self, updatePlot):
        """
        Set the callback functions
        """
        self.updatePlot = updatePlot

    def plotids(self):
        """
        Return the X,Y ids of the selected quantities
        """
        return self.xychoices[self.xSelected], self.xychoices[self.ySelected]

    def updateXPlot(self, newid):
        """
        The x-plot radio buttons have been updated
        """
        self.xSelected = newid
        self.updateChoice()

    def updateYPlot(self, newid):
        """
        The y-plot radio buttons have been updated
        """
        self.ySelected = newid
        self.updateChoice()

    def updateChoice(self):
        # updatePLot was the callback from the main widget
        if self.updatePlot is not None:
            self.updatePlot()



class PlkWidget(QtGui.QWidget):
    """
    The plk-emulator window.

    :param parent:      Parent window
    """

    def __init__(self, parent=None, **kwargs):
        super(PlkWidget, self).__init__(parent, **kwargs)

        self.initPlk()
        self.initPlkLayout()
        self.showVisibleWidgets()

        self.psr = None
        self.parent = parent

    def initPlk(self):
        self.setMinimumSize(650, 550)

        self.plkbox = QtGui.QVBoxLayout()                       # plkbox contains the whole plk widget
        self.xyplotbox = QtGui.QHBoxLayout()                    # plkbox contains the whole plk widget
        self.fitboxesWidget = PlkFitboxesWidget(parent=self)    # Contains all the checkboxes
        self.actionsWidget = PlkActionsWidget(parent=self)

        # We are creating the Figure here, so set the color scheme appropriately
        self.setColorScheme(True)

        # Create the mpl Figure and FigCanvas objects. 
        # 5x4 inches, 100 dots-per-inch
        #
        self.plkDpi = 100
        self.plkFig = Figure((5.0, 4.0), dpi=self.plkDpi)
        self.plkCanvas = FigureCanvas(self.plkFig)
        self.plkCanvas.setParent(self)

        # Since we have only one plot, we can use add_axes 
        # instead of add_subplot, but then the subplot
        # configuration tool in the navigation toolbar wouldn't
        # work.
        #
        self.plkAxes = self.plkFig.add_subplot(111)

        # Done creating the Figure. Restore color scheme to defaults
        self.setColorScheme(False)
        
        # Call-back functions for clicking and key-press.
        self.plkCanvas.mpl_connect('button_press_event', self.canvasClickEvent)
        self.plkCanvas.mpl_connect('key_press_event', self.canvasKeyEvent)

        # Create the navigation toolbar, tied to the canvas
        #
        #self.mpl_toolbar = NavigationToolbar(self.canvas, self.main_frame)

        # Draw an empty graph
        self.drawSomething()

        # Create the XY choice widget
        self.xyChoiceWidget = PlkXYPlotWidget(parent=self)

        # At startup, all the widgets are visible
        self.xyChoiceVisible = True
        self.fitboxVisible = True
        self.actionsVisible = False
        #self.layoutMode = 1         # (0 = none, 1 = all, 2 = only fitboxes, 3 = fit & action)
        self.layoutMode = 4         # (0 = none, 1 = all, 2 = only xy select, 3 = only fit, 4 = xy select & fit)

    def setColorScheme(self, start=True):
        """
        Set the color scheme

        :param start:   When true, save the original scheme, and set to white
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
        self.plkAxes.clear()
        self.plkAxes.grid(True)
        self.plkAxes.set_xlabel('MJD')
        self.plkAxes.set_ylabel('Residual ($\mu$s)')
        self.plkCanvas.draw()
        self.setColorScheme(False)

    def setPulsar(self, psr):
        """
        We've got a new pulsar!
        """
        self.psr = psr

        # Update the fitting checkboxes
        self.fitboxesWidget.setCallbacks(self.fitboxChecked, psr.setpars,
                psr.fitpars, psr.nofitboxpars)
        self.xyChoiceWidget.setCallbacks(self.updatePlot)
        self.actionsWidget.setCallbacks(self.updatePlot, self.reFit)

        # Draw the residuals
        self.xyChoiceWidget.updateChoice()
        # This screws up the show/hide logistics
        #self.show()

    def fitboxChecked(self, parchanged, newstate):
        """
        When a fitbox is (un)checked, this callback function is called

        :param parchanged:  Which parameter has been (un)checked
        :param newstate:    The new state of the checkbox
        """
        self.psr[parchanged].fit = newstate

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

    def initPlkLayout(self):
        """
        Initialise the basic layout of this plk emulator emulator
        """
        # Initialise the plk box
        self.plkbox.addWidget(self.fitboxesWidget)

        self.xyplotbox.addWidget(self.xyChoiceWidget)
        self.xyplotbox.addWidget(self.plkCanvas)

        self.plkbox.addLayout(self.xyplotbox)

        self.plkbox.addWidget(self.actionsWidget)
        self.setLayout(self.plkbox)

    def showVisibleWidgets(self):
        """
        Show the correct widgets in the plk Window
        """
        self.xyChoiceWidget.setVisible(self.xyChoiceVisible)
        self.fitboxesWidget.setVisible(self.fitboxVisible)
        self.actionsWidget.setVisible(self.actionsVisible)


    def updatePlot(self):
        """
        Update the plot/figure
        """
        self.setColorScheme(True)
        self.plkAxes.clear()
        self.plkAxes.grid(True)

        if self.psr is not None:
            # Get a mask for the plotting points
            msk = self.psr.mask('plot')

            #print("Mask has {0} toas".format(np.sum(msk)))

            # Get the IDs of the X and Y axis
            xid, yid = self.xyChoiceWidget.plotids()

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

                self.plotResiduals(xp, yp, yerrp, xlabel, ylabel, self.psr.name)

                if xid in ['mjd', 'year', 'rounded MJD']:
                    self.plotPhaseJumps(self.psr.phasejumps())
            else:
                raise ValueError("Nothing to plot!")

        self.plkCanvas.draw()
        self.setColorScheme(False)


    def plotResiduals(self, x, y, yerr, xlabel, ylabel, title):
        """
        Update the plot, given all the plotting info
        """
        xave = 0.5 * (np.max(x) + np.min(x))
        xmin = xave - 1.05 * (xave - np.min(x))
        xmax = xave + 1.05 * (np.max(x) - xave)
        if yerr is None:
            yave = 0.5 * (np.max(y) + np.min(y))
            ymin = yave - 1.05 * (yave - np.min(y))
            ymax = yave + 1.05 * (np.max(y) - yave)
            self.plkAxes.scatter(x, y, marker='.', color='blue')
        else:
            yave = 0.5 * (np.max(y+yerr) + np.min(y-yerr))
            ymin = yave - 1.05 * (yave - np.min(y-yerr))
            ymax = yave + 1.05 * (np.max(y+yerr) - yave)
            self.plkAxes.errorbar(x, y, yerr=yerr, fmt='.', color='blue')

        self.plkAxes.axis([xmin, xmax, ymin, ymax])
        self.plkAxes.get_xaxis().get_major_formatter().set_useOffset(False)
        self.plkAxes.set_xlabel(xlabel)
        self.plkAxes.set_ylabel(ylabel)
        self.plkAxes.set_title(title, y=1.03)

    def plotPhaseJumps(self, phasejumps):
        """
        Plot the phase jump lines, if we have any
        """
        xmin, xmax, ymin, ymax = self.plkAxes.axis()
        dy = 0.01 * (ymax-ymin)

        if len(phasejumps) > 0:
            phasejumps = np.array(phasejumps)

            for ii in range(len(phasejumps)):
                if phasejumps[ii,1] != 0:
                    self.plkAxes.vlines(phasejumps[ii,0], ymin, ymax,
                            color='darkred', linestyle='--', linewidth=0.5)

                    if phasejumps[ii,1] < 0:
                        jstr = str(phasejumps[ii,1])
                    else:
                        jstr = '+' + str(phasejumps[ii,1])

                    # Print the jump size above the plot
                    ann = self.plkAxes.annotate(jstr, \
                            xy=(phasejumps[ii,0], ymax+dy), xycoords='data', \
                            annotation_clip=False, color='darkred', \
                            size=7.0)
                    

    def setFocusToCanvas(self):
        """
        Set the focus to the plk Canvas
        """
        self.plkCanvas.setFocus()

    def coord2point(self, cx, cy, which='xy'):
        """
        Given data coordinates x and y, obtain the index of the observations
        that is closest to it
        
        :param cx:      x-value of the coordinates
        :param cy:      y-value of the coordinates
        :param which:   which axis to include in distance measure [xy/x/y]
        
        :return:    Index of observation
        """
        ind = None

        if self.psr is not None:
            # Get a mask for the plotting points
            msk = self.psr.mask('plot')

            # Get the IDs of the X and Y axis
            xid, yid = self.xyChoiceWidget.plotids()

            # Retrieve the data
            x, xerr, xlabel = self.psr.data_from_label(xid)
            y, yerr, ylabel = self.psr.data_from_label(yid)

            if np.sum(msk) > 0 and x is not None and y is not None:
                # Obtain the limits
                xmin, xmax, ymin, ymax = self.plkAxes.axis()

                if which == 'xy':
                    dist = ((x[msk]-cx)/(xmax-xmin))**2 + ((y[msk]-cy)/(ymax-ymin))**2
                elif which == 'x':
                    dist = ((x[msk]-cx)/(xmax-xmin))**2
                elif which == 'y':
                    dist = ((y[msk]-cy)/(ymax-ymin))**2
                else:
                    raise ValueError("Value {0} not a valid option for coord2point".format(which))

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
            if 'backspace' in fkey:
                ukey = QtCore.Qt.Key_Backspace

        #if int(e.modifiers()) == (QtCore.Qt.ControlModifier+QtCore.Qt.AltModifier)

        if ukey == QtCore.Qt.Key_Escape:
            if self.parent is None:
                self.close()
            else:
                self.parent.close()
        elif (ukey == ord('M') or ukey == ord('m')) and \
                modifiers == QtCore.Qt.ControlModifier:
            # Change the window
            self.layoutMode = (1+self.layoutMode)%5
            if self.layoutMode == 0:
                self.xyChoiceVisible = False
                self.fitboxVisible = False
                self.actionsVisible = False
            elif self.layoutMode == 1:
                self.xyChoiceVisible = True
                self.fitboxVisible = True
                self.actionsVisible = True
            elif self.layoutMode == 2:
                self.xyChoiceVisible = True
                self.fitboxVisible = False
                self.actionsVisible = False
            elif self.layoutMode == 3:
                self.xyChoiceVisible = False
                self.fitboxVisible = True
                self.actionsVisible = False
            elif self.layoutMode == 4:
                self.xyChoiceVisible = True
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
        elif ukey == ord('+') or ukey == ord('-'):
            # Add/delete a phase jump
            jump = 1
            if ukey == ord('-'):
                jump = -1

            ind = self.coord2point(xpos, ypos, which='x')
            self.psr.add_phasejump(self.psr.stoas[ind], jump)
            self.updatePlot()
        elif ukey == QtCore.Qt.Key_Backspace:
            # Remove all phase jumps
            self.psr.remove_phasejumps()
            self.updatePlot()
        elif ukey == ord('<'):
            # Add a data point to the view on the left
            # TODO: Make this more Pythonic!
            if self.psr['START'].set and self.psr['START'].fit:
                start = self.psr['START'].val
                ltmask = self.psr.stoas < start

                if np.sum(ltmask) > 2:
                    ltind = np.arange(len(self.psr.stoas))[ltmask]
                    lttoas = self.psr.stoas[ltmask]
                    max_ltind = np.argmax(lttoas)

                    # Get maximum of selected TOAs
                    ltmax = ltind[max_ltind]
                    start_max = self.psr.stoas[ltmax]

                    # Get second-highest TOA value
                    ltmask[ltmax] = False
                    ltind = np.arange(len(self.psr.stoas))[ltmask]
                    lttoas = self.psr.stoas[ltmask]
                    max_ltind = np.argmax(lttoas)
                    ltmax = ltind[max_ltind]
                    start_max2 = self.psr.stoas[ltmax]

                    # Set the new START value
                    self.psr['START'].val = 0.5 * (start_max + start_max2)
                elif np.sum(ltmask) == 2:
                    idmin = np.argmin(self.psr.stoas)
                    stmin = self.psr.stoas[idmin]
                    mask = np.ones(len(self.psr.stoas), dtype=np.bool)
                    mask[idmin] = False
                    self.psr['START'].val = 0.5 * \
                            (np.min(self.psr.stoas[mask]) + stmin)
                elif np.sum(ltmask) == 1:
                    self.psr['START'].val = np.min(self.psr.stoas) - 1
                elif np.sum(ltmask) == 0:
                    pass
                self.updatePlot()
        elif ukey == ord('>'):
            # Add a data point to the view on the left
            # TODO: Make this more Pythonic!
            if self.psr['FINISH'].set and self.psr['FINISH'].fit:
                start = self.psr['FINISH'].val
                gtmask = self.psr.stoas > start

                if np.sum(gtmask) > 2:
                    gtind = np.arange(len(self.psr.stoas))[gtmask]
                    gttoas = self.psr.stoas[gtmask]
                    min_gtind = np.argmin(gttoas)

                    # Get maximum of selected TOAs
                    gtmin = gtind[min_gtind]
                    start_min = self.psr.stoas[gtmin]

                    # Get second-highest TOA value
                    gtmask[gtmin] = False
                    gtind = np.arange(len(self.psr.stoas))[gtmask]
                    gttoas = self.psr.stoas[gtmask]
                    min_gtind = np.argmin(gttoas)
                    gtmin = gtind[min_gtind]
                    start_min2 = self.psr.stoas[gtmin]

                    # Set the new FINISH value
                    self.psr['FINISH'].val = 0.5 * (start_min + start_min2)
                elif np.sum(gtmask) == 2:
                    idmax = np.argmax(self.psr.stoas)
                    stmax = self.psr.stoas[idmax]
                    mask = np.ones(len(self.psr.stoas), dtype=np.bool)
                    mask[idmax] = False
                    self.psr['FINISH'].val = 0.5 * \
                            (np.max(self.psr.stoas[mask]) + stmax)
                elif np.sum(gtmask) == 1:
                    self.psr['FINISH'].val = np.max(self.psr.stoas) + 1
                elif np.sum(gtmask) == 0:
                    pass
                self.updatePlot()
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

        #print("PlkWidget: key press: ", ukey, xpos, ypos)

        if not from_canvas:
            if self.parent is not None:
                print("Propagating key press")
                self.parent.keyPressEvent(event)

            super(PlkWidget, self).keyPressEvent(event, **kwargs)

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
        # Callback to the plkWidget
        self.keyPressEvent(event)

