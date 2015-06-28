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
import pysolvepulsar as psp

# All the types of quantities we will be able to plot in the plk widget
plot_labels = ['Residuals', 'mjd', 'year', 'pulse phase', 'orbital phase', \
               'serial', 'day of year', 'frequency', 'TOA error', 'elevation', \
            'rounded MJD', 'sidereal time', 'hour angle', 'para. angle']
nofitboxpars = ['START', 'FINISH', 'POSEPOCH', 'PEPOCH', 'DMEPOCH', \
            'EPHVER', 'TZRMJD', 'TZRFRQ', 'TRES']
plotrangestretch = 1.05

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
        """
        Callback functions
        """
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
    """
    A widget that allows one to select which parameters to fit for
    """

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
        """
        Set the layout of this widget
        """
        # Initialise the layout of the fit-box Widget
        # Initially there are no fitboxes, so just add the hbox
        self.setLayout(self.hbox)


    def setCallbacks(self, boxChecked, setpars, fitpars, nofitbox):
        """
        Set the callback functions

        @param boxChecked:      Callback function, when box is checked
        @param setpars:         psr.parameters(which='set')
        @param fitpars:         psr.parameters(which='fit')
        @param nofitbox:        Which parameters not to have a box for
        """
        self.boxChecked = boxChecked
        self.deleteFitCheckBoxes()
        self.addFitCheckBoxes(setpars, fitpars, nofitbox)

    def addFitCheckBoxes(self, setpars, fitpars, nofitbox):
        """
        Add the fitting checkboxes at the top of the plk Window

        @param setpars:     The parameters that are 'set' (in the model)
        @param fitpars:     The parameters that are currently being fitted for
        @param nofitbox:    The parameters we should skip
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
        """
        Delete all the checkboxes from the Widget. Used when a new pulsar is loaded
        """
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
        """
        This is the signal handler when a checkbox is changed. The changed checkbox
        value will be propagated back to the psr object through callbacks.
        """
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
        self.xychoices = plot_labels
    
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
            if choice.lower() == 'residuals':
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

    @param parent:      Parent window
    """

    def __init__(self, parent=None, **kwargs):
        super(PlkWidget, self).__init__(parent, **kwargs)

        self.pspsr = None         # PulsarSolver object
        self.history = []       # History of previous solution candidates
        self.h_ind = None       # History index
        self.parent = parent

        self.initPlk()
        self.initPlkLayout()
        self.showVisibleWidgets()

    def initPlk(self):
        self.setMinimumSize(650, 550)

        self.plkbox = QtGui.QVBoxLayout()                       # plkbox contains the whole plk widget
        self.xyplotbox = QtGui.QHBoxLayout()                    # plkbox contains the whole plk widget
        #self.fitboxesWidget = PlkFitboxesWidget(parent=self)    # Contains all the checkboxes
        #self.actionsWidget = PlkActionsWidget(parent=self)

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
        self.fitboxVisible = False
        self.actionsVisible = False
        self.layoutMode = 0         # (0 = xy select, 1 = no xy select)

        # Setup the user interface
        self.init_user_interface()

    def init_user_interface(self):
        """Initialize the user interface (zoom, deletions)"""
        # NOTE: Zoom is a general zooming mask. A zoom in one view, can be a
        #       random selection in another.
        if self.pspsr is None:
            self.mask_zoom = None
        else:
            self.mask_zoom = np.ones(self.pspsr.nobs, dtype=np.bool)

        self.show_delete = False        # Whether we show deleted points
        self.mergepatch_lower = None    # Which obs coordinate we are merging
        self.mergepatch_higher = None   # Which obs coordinate we are merging
        self.key_count = 0              # Iteration of the interface
        self.pred_data = None           # Prediction data
        self.pred_realizations = None   # Prediction realizations
        self.pred_stay = 0              # How many more turns before deleting

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
        self.plkAxes.clear()
        self.plkAxes.grid(True)
        self.plkAxes.set_xlabel('MJD')
        self.plkAxes.set_ylabel('Residual ($\mu$s)')
        self.plkCanvas.draw()
        self.setColorScheme(False)

    def setPulsar(self, pspsr, history):
        """
        We've got a new pulsar!
        """
        # Set the pysolvepulsar object
        self.pspsr = pspsr
        self.history = history

        # And set an initial candidate solution
        cand = psp.CandidateSolution()
        cand.set_solution(pspsr.get_prior_values(), *pspsr.get_start_solution())
        self.history.append(cand)
        self.h_ind = 0

        # Setup the user interface
        self.init_user_interface()

        # Update the fitting checkboxes
        # TODO: do not use _psr
        #self.fitboxesWidget.setCallbacks(self.fitboxChecked,
        #    psr._psr.pars(which='set'), psr._psr.pars(which='fit'), nofitboxpars)
        self.xyChoiceWidget.setCallbacks(self.updatePlot)
        #self.actionsWidget.setCallbacks(self.updatePlot, self.reFit)

        # Draw the residuals
        self.xyChoiceWidget.updateChoice()

        # This screws up the show/hide logistics
        #self.show()

    def get_new_parfilename(self):
        """Return a new parfile name"""
        return self.pspsr.name + time.strftime('%Y-%m-%d-%H.%M.%S.par')

    def fitboxChecked(self, parchanged, newstate):
        """
        When a fitbox is (un)checked, this callback function is called

        @param parchanged:  Which parameter has been (un)checked
        @param newstate:    The new state of the checkbox
        """
        #self.psr[parchanged].fit = newstate
        pass

    def reFit(self):
        """
        We need to re-do the fit for this pulsar
        """
        if not self.pspsr is None:
            self.history = self.history[:self.h_ind+1]
            newcand, loglik = self.pspsr.fit_constrained_iterative(
                    self.history[self.h_ind])
            self.history.append(newcand)
            self.h_ind += 1
            self.updatePlot()
    
    def delete_observation(self, ind):
        """Create a new candidate solution in which we delete this point"""
        # Remove the history from here up
        self.history = self.history[:self.h_ind+1]

        # Copy the candidate solution and delete the point
        newcand = psp.CandidateSolution(self.history[self.h_ind])
        newcand.delete_observation(ind)

        # Add the solution
        self.history.append(newcand)
        self.h_ind += 1

    def merge_patches(self, lower, higher):
        """Merge coherence patches, based on these indices

        Merge all coherence patches between these MJDs.
        """
        # Obtain the data, on this scale
        cand = self.history[self.h_ind]
        xid, yid = self.xyChoiceWidget.plotids()
        x, xerr, xlabel = self.pspsr.data_from_label(xid, cand=cand)

        # Find all observations between 'lower' and 'higher'
        inds = np.where(np.logical_and(x >= x[lower], x <= x[higher]))[0]

        if len(inds) == 0:
            # Nothing to merge
            # TODO: Display a warning message?
            return

        # Create a new candidate solution, and remove history after here
        self.history = self.history[:self.h_ind+1]
        newcand = psp.CandidateSolution(self.history[self.h_ind])

        # Find all the patches that belong to these observations
        pinds = []
        for ind in inds:
            pind = newcand.get_patch_from_obsind(ind)
            if pind is not None:
                # This is not a deleted point
                pinds.append(pind)

        # We need a sorted list of unique indices
        upinds = np.unique(pinds)

        # We need the absolute pulse numbers, as determine
        pulse_numbers = self.pspsr.pulsenumbers(newcand)

        # Merge these patches
        for ii, upind in enumerate(upinds[1:]):
            # Indices of upinds change (all -1), after merger
            m1, m2 = upinds[0], upind-ii

            # Join, with no relative phase jump
            newcand.join_patches(m1, m2, pulse_numbers, rpnjump=0)

        # Add the solution
        self.history.append(newcand)
        self.h_ind += 1

    def get_plot_patch_regions(self, x, msk, cand):
        """Given a set of plotting data (x-axis), return patch ids

        Given a set of plotting data for the x-axis, return the patch ids for
        all plotted data that belong to a coherent patch. This allows plotting
        the coherent patches in all plotting scenarios

        :param x:
            The x-axis values to plot (any data)

        :param msk:
            Mask of which observations to plot

        :param cand:
            Candidate solution

        :return:
            List of list of lists, which regions to highlight
            (color, patch, indices)
            Indices of x[msk][isort]
        """
        pvals = np.ones_like(x) * -1

        for pp, patch in enumerate(cand.get_patches()):
            pvals[patch] = pp

        # Sort the selection of points
        isort = np.argsort(x[msk], kind='mergesort')
        xs = x[msk][isort]
        pvs = pvals[msk][isort]
        upvs = np.unique(pvs)

        #bucket_ind = [[[]]]       # color, patch, indices
        bucket_ind = []       # color, patch, indices
        for cc, up in enumerate(upvs):
            # Start a new color
            bucket_ind.append([[]])
            uinds = np.where(pvs == up)[0]

            bucket_ind[cc][-1].append(uinds[0])
            for jj in uinds[1:]:
                if jj == bucket_ind[cc][-1][-1] + 1:
                    # Next element, add to patch
                    bucket_ind[cc][-1].append(jj)
                else:
                    # Elements are skipped. Start new patch
                    bucket_ind[cc].append([jj])

        return bucket_ind


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
        #self.plkbox.addWidget(self.fitboxesWidget)

        self.xyplotbox.addWidget(self.xyChoiceWidget)
        self.xyplotbox.addWidget(self.plkCanvas)

        self.plkbox.addLayout(self.xyplotbox)

        #self.plkbox.addWidget(self.actionsWidget)
        self.setLayout(self.plkbox)

    def showVisibleWidgets(self):
        """
        Show the correct widgets in the plk Window
        """
        self.xyChoiceWidget.setVisible(self.xyChoiceVisible)
        #self.fitboxesWidget.setVisible(self.fitboxVisible)
        #self.actionsWidget.setVisible(self.actionsVisible)

    def get_plotting_mask(self, cand):
        """Return a mask we use for plotting"""
        mask_delete = np.ones(len(self.mask_zoom), dtype=np.bool)
        mask_delete[cand.obs_inds] = False
        return self.mask_zoom.copy() if self.show_delete else \
                np.logical_and(self.mask_zoom, np.logical_not(mask_delete))

    def get_screen_limits(self, x, y, yerr=None):
        """Return the screen limits, based on the plotting data"""
        xave = 0.5 * (np.max(x) + np.min(x))
        xmin = xave - plotrangestretch * (xave - np.min(x))
        xmax = xave + plotrangestretch * (np.max(x) - xave)
        if yerr is None:
            yave = 0.5 * (np.max(y) + np.min(y))
            ymin = yave - plotrangestretch * (yave - np.min(y))
            ymax = yave + plotrangestretch * (np.max(y) - yave)
        else:
            yave = 0.5 * (np.max(y+yerr) + np.min(y-yerr))
            ymin = yave - plotrangestretch * (yave - np.min(y-yerr))
            ymax = yave + plotrangestretch * (np.max(y+yerr) - yave)

        return (xmin, xmax, ymin, ymax)

    def updatePlot(self):
        """
        Update the plot/figure
        """
        self.setColorScheme(True)
        self.plkAxes.clear()
        self.plkAxes.grid(True)

        if self.pspsr is not None:
            # The candidate solution we are plotting now
            cand = self.history[self.h_ind]

            # Get a mask for the plotting points
            msk = self.get_plotting_mask(cand)

            # Get the IDs of the X and Y axis
            xid, yid = self.xyChoiceWidget.plotids()

            # Retrieve the data
            x, xerr, xlabel = self.pspsr.data_from_label(xid, cand=cand)
            y, yerr, ylabel = self.pspsr.data_from_label(yid, cand=cand)

            if x is not None and y is not None and np.sum(msk) > 0:
                xp = x[msk]
                yp = y[msk]

                if yerr is not None:
                    yerrp = yerr[msk]
                else:
                    yerrp = None

                self.plotResiduals(xp, yp, yerrp, xlabel, ylabel, self.pspsr.name)
                self.plotCoherentPatches(x, msk, yp, yerrp, cand)
                if self.pred_data is not None and \
                        xid == 'mjd' and yid in ['Residuals', 'pulse phase']:
                    #self.plotPrediction(*self.pred_data)
                    F0 = self.pspsr._psr['F0'].val
                    bound, mult = (1.0/F0, 1.0) if yid == 'Residuals' else (1.0, F0)
                           
                    self.plotPredictionRealizations(yp*mult, self.pred_realizations,
                            bound, *self.pred_data)

                    self.pred_stay -= 1
                    if self.pred_stay == 0:
                        self.pred_data = None
                        self.pred_realizations = None
            else:
                raise ValueError("Nothing to plot!")

        self.plkCanvas.draw()
        self.setColorScheme(False)

    def plotCoherentPatches(self, x, msk, y, yerr, cand):
        """Plot the coherence patches as colored regions

        Plot the coherence patches as colored regions.

        :param x:
            All the x-axis values

        :param msk:
            Mask of x-values, which we will plot

        :param cand:
            Candidate solution
        """
        bucket = self.get_plot_patch_regions(x, msk, cand)
        isort = np.argsort(x[msk], kind='mergesort')
        xs = x[msk][isort]

        # Screen limits
        xmin, xmax, ymin, ymax = self.get_screen_limits(x[msk], y, yerr)

        colors = ['b', 'r', 'g', 'c', 'y', 'k']
        ncols = len(colors)

        ncol = 0
        for bucket_col in [x for x in bucket if len(x) > 1 or len(x[0]) > 1]:
            # Have a patch with more than one element, or multiple patches
            for patch in bucket_col:
                if patch[0] == 0: #x[patch[0]] == np.min(x[msk]):
                    # Left-val edge of screen
                    x_left = xmin
                else:
                    x_left = 0.5 * (xs[patch[0]] + xs[patch[0]-1])

                if patch[-1] == len(xs)-1:#x[patch[-1]] == np.max(x[msk]):
                    # Rigth-val edge of screen
                    x_right = xmax
                else:
                    x_right = 0.5 * (xs[patch[-1]] + xs[patch[-1]+1])

                # Draw the patch
                self.plkAxes.fill_between([x_left, x_right], ymin, ymax, \
                        facecolor=colors[ncol % ncols], alpha=0.1)

            ncol += 1


    def plotResiduals(self, x, y, yerr, xlabel, ylabel, title):
        """
        Update the plot, given all the plotting info
        """
        xmin, xmax, ymin, ymax = self.get_screen_limits(x, y, yerr)

        if yerr is None:
            self.plkAxes.scatter(x, y, marker='.', color='blue')
        else:
            self.plkAxes.errorbar(x, y, yerr=yerr, fmt='.', color='blue')

        self.plkAxes.axis([xmin, xmax, ymin, ymax])
        self.plkAxes.get_xaxis().get_major_formatter().set_useOffset(False)
        self.plkAxes.set_xlabel(xlabel)
        self.plkAxes.set_ylabel(ylabel)
        self.plkAxes.set_title(title, y=1.03)

    def plotPrediction(self, t, dt_r, dt_stdrp):
        P0 = 1.0 / self.pspsr._psr['F0'].val
        self.plkAxes.plot(t, dt_r, 'k--', linewidth=2.0)
        rmin, rmax = dt_r-dt_stdrp, dt_r+dt_stdrp
        rmin[rmin < -0.5*P0] = -0.5*P0
        rmax[rmin > 0.5*P0] = 0.5*P0
        self.plkAxes.fill_between(t, rmin, rmax, facecolor='k', alpha=0.3)

    def plotPredictionRealizations(self, yp, mr, bound, t, dt_r, dt_stdrp):
        #P0 = 1.0 / self.pspsr._psr['F0'].val
        self.plkAxes.plot(t, dt_r, 'k--', linewidth=2.0)
        rmin, rmax = dt_r-dt_stdrp, dt_r+dt_stdrp
        pmin, pmax = min(-0.5*bound, np.min(yp)), max(0.5*bound, np.max(yp))
        rmin[rmin < pmin] = pmin
        rmax[rmin > pmax] = pmax
        self.plkAxes.plot(t, rmin, 'k-', linewidth=1.2)
        self.plkAxes.plot(t, rmax, 'k-', linewidth=1.2)
        self.plkAxes.fill_between(t, rmin, rmax, facecolor='k', alpha=0.1)
        for ii in range(mr.shape[1]):
            r = mr[:,ii]
            r[r < -0.5*bound] = -0.5*bound
            r[r > 0.5*bound] = 0.5*bound
            self.plkAxes.plot(t, (dt_r+r), c='darkred', linestyle='-',
                    linewidth=0.5, alpha=0.3)

        #self.plkAxes.plot(t, dt_r, 'k--', linewidth=2.0)
        #rmin, rmax = dt_r-dt_stdrp, dt_r+dt_stdrp
        #rmin[rmin < -0.5*bound] = -0.5*bound
        #rmax[rmin > 0.5*bound] = 0.5*bound
        #self.plkAxes.fill_between(t, rmin, rmax, facecolor='k', alpha=0.3)

    def plotPhaseJumps_deprecated(self, phasejumps):
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

    def coord2point(self, cx, cy, which='xy', mode='both'):
        """
        Given data coordinates x and y, obtain the index of the observations
        that is closest to it
        
        @param cx:      x-value of the coordinates
        @param cy:      y-value of the coordinates
        @param which:   which axis to include in distance measure [xy/x/y]
        @param mode:    From which side to include (both/lower/higher)
                        (used when which='x' or which='y'
        
        @return:    Index of observation
        """
        ind = None

        if self.pspsr is not None:
            # The candidate solution we are plotting now
            cand = self.history[self.h_ind]

            # Get a mask for the plotting points
            msk = self.get_plotting_mask(cand)

            # Get the IDs of the X and Y axis
            xid, yid = self.xyChoiceWidget.plotids()

            # Retrieve the data
            x, xerr, xlabel = self.pspsr.data_from_label(xid, cand=cand)
            y, yerr, ylabel = self.pspsr.data_from_label(yid, cand=cand)

            if np.sum(msk) > 0 and x is not None and y is not None:
                # Obtain the limits
                xmin, xmax, ymin, ymax = self.plkAxes.axis()

                if which == 'xy':
                    dist = ((x[msk]-cx)/(xmax-xmin))**2 + ((y[msk]-cy)/(ymax-ymin))**2
                elif which == 'x' and mode == 'both':
                    dist = ((x[msk]-cx)/(xmax-xmin))**2
                elif which == 'y' and mode == 'both':
                    dist = ((y[msk]-cy)/(ymax-ymin))**2
                elif which == 'x' and mode == 'lower':
                    dist = ((x[msk]-cx)/(xmax-xmin))**2
                    dist[x[msk]-cx >= 0] = np.inf
                    if np.all(dist == np.inf):
                        dist[np.argmin(x[msk])] = 0.0
                elif which == 'x' and mode == 'higher':
                    dist = ((x[msk]-cx)/(xmax-xmin))**2
                    dist[x[msk]-cx <= 0] = np.inf
                    if np.all(dist == np.inf):
                        dist[np.argmax(x[msk])] = 0.0
                elif which == 'y' and mode == 'lower':
                    dist = ((y[msk]-cy)/(ymax-ymin))**2
                    dist[y[msk]-cy >= 0] = np.inf
                    if np.all(dist == np.inf):
                        dist[np.argmin(x[msk])] = 0.0
                elif which == 'y' and mode == 'higher':
                    dist = ((y[msk]-cy)/(ymax-ymin))**2
                    dist[y[msk]-cy <= 0] = np.inf
                    if np.all(dist == np.inf):
                        dist[np.argmax(x[msk])] = 0.0
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
                modifiers += QtCore.Qt.AltModifier
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
        if (ukey == ord('z') or ukey == ord('Z')) and \
                (modifiers == QtCore.Qt.ControlModifier or \
                modifiers == QtCore.Qt.MetaModifier):
            # TODO: How do I really catch the MetaModifier? On OSX, I can't get
            #       the command key
            if self.h_ind > 0:
                self.h_ind -= 1

            self.updatePlot()
        if (ukey == ord('r') or ukey == ord('R')) and \
                (modifiers == QtCore.Qt.ControlModifier or \
                modifiers == QtCore.Qt.MetaModifier):
            # TODO: How do I really catch the MetaModifier? On OSX, I can't get
            #       the command key
            if self.h_ind + 1 < len(self.history):
                self.h_ind += 1

            self.updatePlot()
        elif (ukey == ord('M') or ukey == ord('m')) and \
                modifiers == QtCore.Qt.ControlModifier:
                #(modifiers == QtCore.Qt.ControlModifier or \
                #modifiers == QtCore.Qt.MetaModifier):
            # Change the window
            self.layoutMode = (1+self.layoutMode)%2
            if self.layoutMode == 0:
                self.xyChoiceVisible = True
                self.fitboxVisible = False
                self.actionsVisible = False
            elif self.layoutMode == 1:
                self.xyChoiceVisible = False
                self.fitboxVisible = False
                self.actionsVisible = False
            self.showVisibleWidgets()
        elif ukey == ord('s'):
            # Get the x-value data
            cand = self.history[self.h_ind]
            xid, yid = self.xyChoiceWidget.plotids()
            x, xerr, xlabel = self.pspsr.data_from_label(xid, cand=cand)

            # Get the index of the start-point
            ind = self.coord2point(xpos, ypos, which='x', mode='higher')

            # Mask everything before it (not the point itself)
            self.mask_zoom[x < x[ind]] = False

            self.updatePlot()
        elif ukey == ord('f'):
            # Get the x-value data
            cand = self.history[self.h_ind]
            xid, yid = self.xyChoiceWidget.plotids()
            x, xerr, xlabel = self.pspsr.data_from_label(xid, cand=cand)

            # Get the index of the start-point
            ind = self.coord2point(xpos, ypos, which='x', mode='lower')

            # Mask everything after it (not the point itself)
            self.mask_zoom[x > x[ind]] = False

            self.updatePlot()
        elif ukey == ord('u'):
            # Unzoom
            self.mask_zoom[:] = True
            self.updatePlot()
        elif ukey == ord('d'):
            # Delete data point
            ind = self.coord2point(xpos, ypos)
            self.delete_observation(ind)

            self.updatePlot()
        elif ukey == ord('['):
            # Start of a coherence-patch merge
            # Get the x-value data
            cand = self.history[self.h_ind]
            xid, yid = self.xyChoiceWidget.plotids()
            x, xerr, xlabel = self.pspsr.data_from_label(xid, cand=cand)

            # Get the index of the start-point
            ind = self.coord2point(xpos, ypos, which='x', mode='higher')

            if self.mergepatch_higher is not None:
                self.merge_patches(ind, self.mergepatch_higher)

                # Reset the memory
                self.mergepatch_lower = None
                self.mergepatch_higher = None
            else:
                self.mergepatch_lower = ind

            self.updatePlot()
        elif ukey == ord(']'):
            # End of a coherence-patch merge
            # Get the x-value data
            cand = self.history[self.h_ind]
            xid, yid = self.xyChoiceWidget.plotids()
            x, xerr, xlabel = self.pspsr.data_from_label(xid, cand=cand)

            # Get the index of the start-point
            ind = self.coord2point(xpos, ypos, which='x', mode='lower')

            if self.mergepatch_lower is not None:
                self.merge_patches(self.mergepatch_lower, ind)

                # Reset the memory
                self.mergepatch_lower = None
                self.mergepatch_higher = None
            else:
                self.mergepatch_higher = ind

            self.updatePlot()
        elif ukey == ord('c') or ukey == ord('C'):
            # Split a coherence patch at this point
            self.history = self.history[:self.h_ind+1]
            newcand = psp.CandidateSolution(self.history[self.h_ind])
            newcand = self.history[self.h_ind]
            ind = self.coord2point(xpos, ypos, which='x', mode='higher')
            xid, yid = self.xyChoiceWidget.plotids()
            x, xerr, xlabel = self.pspsr.data_from_label(xid, cand=newcand)
            pind = newcand.get_patch_from_obsind(ind)
            if pind is not None:
                patch = newcand._patches[pind]
                msk = (x[patch] >= x[ind])
                newcand.split_patch(pind, msk)

                self.history.append(newcand)
                self.h_ind += 1

            self.updatePlot()
        elif ukey == ord('e') or ukey == ord('E'):
            # Do the extrapolation
            # First obtain the plotting ranges etc.
            cand = self.history[self.h_ind]
            xid, yid = self.xyChoiceWidget.plotids()
            if xid in ['mjd'] and yid in ['Residuals', 'pulse phase']:
                # Get the screen layout
                x, xerr, xlabel = self.pspsr.data_from_label(xid, cand=cand)
                y, yerr, ylabel = self.pspsr.data_from_label(yid, cand=cand)
                msk = self.get_plotting_mask(cand)
                xmin, xmax, ymin, ymax = self.get_screen_limits(x[msk],
                        y[msk], yerr[msk])

                # Obtain the patch from which we'll extrapolate
                ind = self.coord2point(xpos, ypos, which='xy')
                pind = cand.get_patch_from_obsind(ind)
                if pind is not None:
                    # Perform the linear least-squares extrapolation
                    dd = self.pspsr.perform_linear_least_squares_fit_old(cand,
                            fitpatch=pind)

                    mult = 1.0 if yid == 'Residuals' else \
                           self.pspsr._psr['F0'].val

                    npobs = self.pspsr.predpsr_nobs
                    obstimes = np.linspace(xmin, xmax, npobs)
                    dt_r, dt_stdrp = self.pspsr.get_mock_prediction(obstimes, dd)
                    self.pred_data = (obstimes, dt_r*mult, dt_stdrp*mult)
                    self.pred_realizations = self.pspsr.get_mock_realizations(\
                            obstimes, dd)[1] * mult
                    self.pred_stay = 2

            self.updatePlot()
        elif ukey == ord('+') or ukey == ord('-'):
            # Add/delete a phase jump
            rpnjump = 1 if ukey == ord('-') else -1

            self.history = self.history[:self.h_ind+1]
            newcand = psp.CandidateSolution(self.history[self.h_ind])
            #newcand = self.history[self.h_ind]
            ind = self.coord2point(xpos, ypos, which='x', mode='higher')
            xid, yid = self.xyChoiceWidget.plotids()
            x, xerr, xlabel = self.pspsr.data_from_label(xid, cand=newcand)
            pind = newcand.get_patch_from_obsind(ind)
            if pind is not None:
                patch = newcand._patches[pind]
                msk = (x[patch] >= x[ind])

                newcand.add_rpn_jump(pind, msk, rpnjump)

                self.history.append(newcand)
                self.h_ind += 1

            self.updatePlot()
            """
            jump = 1
            if ukey == ord('-'):
                jump = -1

            ind = self.coord2point(xpos, ypos, which='x')
            self.psr.add_phasejump(self.psr.stoas[ind], jump)
            self.updatePlot()
            """
        elif ukey == QtCore.Qt.Key_Backspace:
            # Remove all phase jumps
            # Not necessary anymore
            pass
        elif ukey == ord('P'):
            # Write a new parfile
            parfilename = self.get_new_parfilename()
            cand = self.history[self.h_ind]

            dd = self.pspsr.perform_linear_least_squares_fit_old(cand)
            self.pspsr.savepar(parfilename, dd)
        elif ukey == ord('<'):
            """
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
            """
            self.updatePlot()
        elif ukey == ord('>'):
            """
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
            """
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
        self.key_count += 1


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

