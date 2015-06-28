#!/usr/bin/python
# -*- coding: utf-8 -*-
# vim: tabstop=4:softtabstop=4:shiftwidth=4:expandtab
"""
qtip: Qt interactive interface for PTA data analysis tools

"""


from __future__ import print_function
from __future__ import division
import os, sys

# Importing all the stuff for the IPython console widget
from IPython.qt.console.rich_ipython_widget import RichIPythonWidget
from IPython.qt.inprocess import QtInProcessKernelManager
from IPython.lib import guisupport

from PyQt4 import QtGui, QtCore

# Advanced command-line option parsing
import optparse

# Numpy etc.
import numpy as np
import time
import matplotlib
import tempfile

import qtpulsar as qp
import constants

from opensomething import OpenSomethingWidget
from plk import PlkWidget
from binary import BinaryWidget

# The startup banner
QtipBanner = """Qtip python console, by Rutger van Haasteren
Console powered by IPython
Type "copyright", "credits" or "license" for more information.

?         -> Introduction and overview of IPython's features.
%quickref -> Quick reference.
help      -> Python's own help system.
object?   -> Details about 'object', use 'object??' for extra details.
%guiref   -> A brief reference about the graphical user interface.

import numpy as np, matplotlib.pyplot as plt, qtpulsar as qp
"""




class QtipWindow(QtGui.QMainWindow):
    """
    Main Qtip window

    Note, is the main window now, but the content will later be moved to a
    libstempo tab, as part of the Piccard suite
    """
    
    def __init__(self, parent=None, engine='libstempo', \
            parfile=None, timfile=None, perfile=None, **kwargs):
        super(QtipWindow, self).__init__(parent)
        self.setWindowTitle('QtIpython interface to PINT/libstempo')

        # Initialise basic gui elements
        self.initUI()

        # Start the embedded IPython kernel
        self.createIPythonKernel()

        # Create the display widgets
        self.createBinaryWidget()
        self.createPlkWidget()
        self.createIPythonWidget()
        self.createOpenSomethingWidget()

        # Position the widgets
        self.initQtipLayout()
        self.setQtipLayout(whichWidget='binary',
                showIPython=False, firsttime=True)

        self.pref_engine = engine
        if True:
            # We are still in MAJOR testing mode, so open a test-pulsar right away
            # (delete this line when going into production)
            if parfile is None or timfile is None:
                testpulsar = True
            else:
                testpulsar = False

            # Also open the Binary Widget
            self.requestOpenBinary(testpulsar=True)

            # Are we going to open plk straight away?
            self.requestOpenPlk(testpulsar=testpulsar, parfilename=parfile, \
                    timfilename=timfile, engine=self.pref_engine)
        else:
            if perfile is None:
                testpulsar = True
            else:
                testpulsar = False

                if parfile is None:
                    parfile = ""

            self.requestOpenBinary(testpulsar=testpulsar, parfilename=parfile, \
                    perfilename=perfile)

        self.show()

    def __del__(self):
        pass

    def onAbout(self):
        msg = """ A plk emulator, written in Python. Powered by PyQt, matplotlib, libstempo, and IPython:
        """
        QtGui.QMessageBox.about(self, "About the demo", msg.strip())

    def initUI(self):
        """
        Initialise the user-interface elements
        """
        # Create the main-frame widget, and the layout
        self.mainFrame = QtGui.QWidget()
        self.setCentralWidget(self.mainFrame)
        self.hbox = QtGui.QHBoxLayout()     # HBox contains all widgets

        # Create the menu action items
        self.openParTimAction = QtGui.QAction('&Open par/tim', self)        
        self.openParTimAction.setShortcut('Ctrl+O')
        self.openParTimAction.setStatusTip('Open par/tim')
        self.openParTimAction.triggered.connect(self.openParTim)

        self.openParPerAction = QtGui.QAction('Open &par/bestprof', self)
        self.openParPerAction.setShortcut('Ctrl+G')
        self.openParPerAction.setStatusTip('Open par/bestprof files')
        self.openParPerAction.triggered.connect(self.openParPer)

        self.openPerAction = QtGui.QAction('Open &bestprof', self)
        self.openPerAction.setShortcut('Ctrl+H')
        self.openPerAction.setStatusTip('Open bestprof files')
        self.openPerAction.triggered.connect(self.openPer)

        self.exitAction = QtGui.QAction(QtGui.QIcon('exit.png'), '&Exit', self)        
        self.exitAction.setShortcut('Ctrl+Q')
        self.exitAction.setStatusTip('Exit application')
        self.exitAction.triggered.connect(self.close)

        self.toggleBinaryAction = QtGui.QAction('&Binary', self)        
        self.toggleBinaryAction.setShortcut('Ctrl+B')
        self.toggleBinaryAction.setStatusTip('Toggle binary widget')
        self.toggleBinaryAction.triggered.connect(self.toggleBinary)

        self.togglePlkAction = QtGui.QAction('&Plk', self)        
        self.togglePlkAction.setShortcut('Ctrl+P')
        self.togglePlkAction.setStatusTip('Toggle plk widget')
        self.togglePlkAction.triggered.connect(self.togglePlk)

        self.toggleIPythonAction = QtGui.QAction('&IPython', self)        
        self.toggleIPythonAction.setShortcut('Ctrl+I')
        self.toggleIPythonAction.setStatusTip('Toggle IPython')
        self.toggleIPythonAction.triggered.connect(self.toggleIPython)

        self.aboutAction = QtGui.QAction('&About', self)        
        self.aboutAction.setShortcut('Ctrl+A')
        self.aboutAction.setStatusTip('About Qtip')
        self.aboutAction.triggered.connect(self.onAbout)

        self.theStatusBar = QtGui.QStatusBar()
        #self.statusBar()
        self.setStatusBar(self.theStatusBar)

        self.engine_label = QtGui.QLabel("Tempo2")
        self.engine_label.setFrameStyle( QtGui.QFrame.Sunken|QtGui.QFrame.Panel)
        self.engine_label.setLineWidth(4)
        self.engine_label.setMidLineWidth(4)
        self.engine_label.setStyleSheet("QLabel{color:black;background-color:red}")
        self.theStatusBar.addPermanentWidget(self.engine_label)

        # On OSX, make sure the menu can be displayed (in the window itself)
        if sys.platform == 'darwin':
            # On OSX, the menubar is usually on the top of the screen, not in
            # the window. To make it in the window:
            QtGui.qt_mac_set_native_menubar(False) 

            # Otherwise, if we'd like to get the system menubar at the top, then
            # we need another menubar object, not self.menuBar as below. In that
            # case, use:
            # self.menubar = QtGui.QMenuBar()
            # TODO: Somehow this does not work. Per-window one does though

        # Create the menu
        self.menubar = self.menuBar()
        self.fileMenu = self.menubar.addMenu('&File')
        self.fileMenu.addAction(self.openParTimAction)
        self.fileMenu.addAction(self.openParPerAction)
        self.fileMenu.addAction(self.openPerAction)
        self.fileMenu.addAction(self.exitAction)
        self.viewMenu = self.menubar.addMenu('&View')
        self.viewMenu.addAction(self.toggleBinaryAction)
        self.viewMenu.addAction(self.togglePlkAction)
        self.viewMenu.addAction(self.toggleIPythonAction)
        self.helpMenu = self.menubar.addMenu('&Help')
        self.helpMenu.addAction(self.aboutAction)

        # What is the status quo of the user interface?
        self.showIPython = False
        self.whichWidget = 'None'
        self.prevShowIPython = None
        self.prevWhichWidget = 'None'

    def createIPythonKernel(self):
        """
        Create the IPython Kernel
        """
        # Create an in-process kernel
        self.kernelManager = QtInProcessKernelManager()
        self.kernelManager.start_kernel()
        self.kernel = self.kernelManager.kernel

        self.kernelClient = self.kernelManager.client()
        self.kernelClient.start_channels()

        self.kernel.shell.enable_matplotlib(gui='inline')

        # Load the necessary packages in the embedded kernel
        cell = "import numpy as np, matplotlib.pyplot as plt, qtpulsar as qp, libfitorbit as lo"
        self.kernel.shell.run_cell(cell, store_history=False)

        # Set the in-kernel matplotlib color scheme to black.
        self.setMplColorScheme('black')     # Outside as well (do we need this?)
        self.kernel.shell.run_cell(constants.matplotlib_rc_cell_black,
                store_history=False)

    def createIPythonWidget(self):
        """
        Create the IPython widget
        """
        self.consoleWidget = RichIPythonWidget()
        #self.consoleWidget.setMinimumSize(600, 550)
        self.consoleWidget.banner = QtipBanner
        self.consoleWidget.kernel_manager = self.kernelManager
        self.consoleWidget.kernel_client = self.kernelClient
        self.consoleWidget.exit_requested.connect(self.toggleIPython)
        self.consoleWidget.set_default_style(colors='linux')
        self.consoleWidget.hide()

        # Register a call-back function for the IPython shell. This one is
        # executed insite the child-kernel.
        #self.kernel.shell.register_post_execute(self.postExecute)
        #
        # In IPython >= 2, we can use the event register
        # Events: post_run_cell, pre_run_cell, etc...`
        self.kernel.shell.events.register('pre_execute', self.preExecute)
        self.kernel.shell.events.register('post_execute', self.postExecute)
        self.kernel.shell.events.register('post_run_cell', self.postRunCell)


    def createOpenSomethingWidget(self):
        """
        Create the OpenSomething widget. Do not add it to the layout yet

        TODO:   This widget will become the first main widget to see. At the
                moment, however, we're avoiding it for the sake of testing
                purposes
        """
        # TODO: This widget is not really used at the moment
        self.openSomethingWidget = OpenSomethingWidget(parent=self.mainFrame, \
                openFile=self.requestOpenPlk)
        self.openSomethingWidget.hide()

    def createPlkWidget(self):
        """
        Create the Plk widget
        """
        self.plkWidget = PlkWidget(parent=self.mainFrame)
        self.plkWidget.hide()

    def createBinaryWidget(self):
        """
        Create the binary model widget
        """
        self.binaryWidget = BinaryWidget(parent=self.mainFrame)
        self.binaryWidget.hide()

    def toggleIPython(self):
        """
        Toggle the IPython widget on or off
        """
        self.setQtipLayout(showIPython = not self.showIPython)

    def toggleBinary(self):
        """
        Toggle the binary model widget on or off
        """
        self.setQtipLayout(whichWidget='binary')

    def togglePlk(self):
        """
        Toggle the plk widget on or off
        """
        self.setQtipLayout(whichWidget='plk')


    def initQtipLayout(self):
        """
        Initialise the Qtip layout
        """
        self.hbox.addWidget(self.openSomethingWidget)
        self.hbox.addWidget(self.plkWidget)
        self.hbox.addWidget(self.binaryWidget)
        self.hbox.addStretch(1)
        self.hbox.addWidget(self.consoleWidget)
        self.mainFrame.setLayout(self.hbox)

    def hideAllWidgets(self):
        """
        Hide all widgets of the mainFrame
        """
        # Remove all widgets from the main window
        # ???
        """
        while self.hbox.count():
            item = self.hbox.takeAt(0)
            if isinstance(item, QtGui.QWidgetItem):
                #item.widget().deleteLater()
                item.widget().hide()
            elif isinstance(item, QtGui.QSpacerItem):
                #self.hbox.removeItem(item)
                pass
            else:
                #fcbox.clearLayout(item.layout())
                #self.hbox.removeItem(item)
                pass
        """
        self.openSomethingWidget.hide()
        self.plkWidget.hide()
        self.binaryWidget.hide()
        self.consoleWidget.hide()

    def showVisibleWidgets(self):
        """
        Show the correct widgets in the mainFrame
        """
        # Add the widgets we need
        if self.whichWidget.lower() == 'opensomething':
            self.openSomethingWidget.show()
        elif self.whichWidget.lower() == 'plk':
            self.plkWidget.show()
        elif self.whichWidget.lower() == 'piccard':
            pass
        elif self.whichWidget.lower() == 'binary':
            self.binaryWidget.show()

        if self.showIPython:
            self.consoleWidget.show()
        else:
            pass

        if self.whichWidget.lower() == 'plk' and not self.showIPython:
            self.plkWidget.setFocusToCanvas()
        elif self.whichWidget.lower() == 'binary' and not self.showIPython:
            self.binaryWidget.setFocusToCanvas()
        #elif self.showIPython:
        #    self.consoleWidget.setFocus()

    def setQtipLayout(self, whichWidget=None, showIPython=None, firsttime=False):
        """
        Given which widgets to show, display the right widgets and hide the rest

        @param whichWidget:     Which widget to show
        @param showIPython:     Whether to show the IPython console
        """
        if not whichWidget is None:
            self.whichWidget = whichWidget
        if not showIPython is None:
            self.showIPython = showIPython

        # After hiding the widgets, wait 25 (or 0?) miliseconds before showing them again
        self.hideAllWidgets()
        QtCore.QTimer.singleShot(0, self.showVisibleWidgets)

        self.prevWhichWidget = self.whichWidget

        if self.showIPython != self.prevShowIPython:
            # IPython has been toggled
            self.prevShowIPython = self.showIPython
            if self.showIPython:
                self.resize(1350, 550)
                self.mainFrame.resize(1350, 550)
            else:
                self.resize(650, 550)
                self.mainFrame.resize(650, 550)

        if firsttime:
            # Set position slightly more to the left of the screen, so we can
            # still open IPython
            self.move(50, 100)

        self.mainFrame.setLayout(self.hbox)
        self.mainFrame.show()

    def requestOpenBinary(self, parfilename=None, perfilename=None, \
            testpulsar=False):
        """
        Request to open a file in the binary widget

        @param parfilename:     The parfile to open. If None, ask the user
        @param perfilename:     The per/bestprof file to open. If None, ask user
        """
        self.setQtipLayout(whichWidget='binary', showIPython=self.showIPython)

        if parfilename is None and not testpulsar:
            parfilename = QtGui.QFileDialog.getOpenFileName(self, 'Open par-file', '~/')
        elif parfilename == "":
            # We do not need to load a par file
            parfilename = None

        if perfilename is None and not testpulsar:
            perfilename = QtGui.QFileDialog.getOpenFileName(self, 'Open per/bestprof file', '~/')

        # Load the pulsar
        self.openBinaryPulsar(parfilename, perfilename, testpulsar=testpulsar)

    def requestOpenPlk(self, parfilename=None, timfilename=None, \
            testpulsar=False, engine='libstempo'):
        """
        Request to open a file in the plk widget

        @param parfilename:     The parfile to open. If none, ask the user
        @param timfilename:     The timfile to open. If none, ask the user
        """
        self.setQtipLayout(whichWidget='plk', showIPython=self.showIPython)

        if parfilename is None and not testpulsar:
            parfilename = QtGui.QFileDialog.getOpenFileName(self, 'Open par-file', '~/')

        if timfilename is None and not testpulsar:
            timfilename = QtGui.QFileDialog.getOpenFileName(self, 'Open tim-file', '~/')

        # Load the pulsar
        self.openPlkPulsar(parfilename, timfilename, engine=engine, \
                testpulsar=testpulsar)

    def setMplColorScheme(self, scheme):
        """
        Set the matplotlib color scheme

        @param scheme:  'black'/'white', the color scheme
        """

        # Obtain the Widget background color
        color = self.palette().color(QtGui.QPalette.Window)
        r, g, b = color.red(), color.green(), color.blue()
        rgbcolor = (r/255.0, g/255.0, b/255.0)

        if scheme == 'white':
            rcP = constants.mpl_rcParams_white

            rcP['axes.facecolor'] = rgbcolor
            rcP['figure.facecolor'] = rgbcolor
            rcP['figure.edgecolor'] = rgbcolor
            rcP['savefig.facecolor'] = rgbcolor
            rcP['savefig.edgecolor'] = rgbcolor
        elif scheme == 'black':
            rcP = constants.mpl_rcParams_black

        for key, value in rcP.iteritems():
            matplotlib.rcParams[key] = value


    def openParTim(self):
        """
        Open a par-file and a tim-file
        """
        # TODO: obtain the engine from elsewhere
        #engine='libstempo'

        # Ask the user for a par and tim file, and open these with libstempo
        parfilename = QtGui.QFileDialog.getOpenFileName(self, 'Open par-file', '~/')
        timfilename = QtGui.QFileDialog.getOpenFileName(self, 'Open tim-file', '~/')

        # Load the pulsar
        self.openPlkPulsar(parfilename, timfilename, engine=self.pref_engine)

    def openParPer(self):
        """
        Open a par-file and a per/bestprof file
        """
        # Ask the user for a par and tim file, and open these with libstempo
        parfilename = QtGui.QFileDialog.getOpenFileName(self, 'Open par-file', '~/')
        perfilename = QtGui.QFileDialog.getOpenFileName(self, 'Open per/bestprof-file', '~/')

        # Load the pulsar
        self.openBinaryPulsar(parfilename=parfilename, \
                perfilename=perfilename)

    def openPer(self):
        """
        Open a per/bestprof file
        """
        # Ask the user for a par and tim file, and open these with libstempo
        perfilename = QtGui.QFileDialog.getOpenFileName(self, 'Open per/bestprof-file', '~/')

        # Load the pulsar
        self.openBinaryPulsar(parfilename=None, perfilename=perfilename)

    def openPlkPulsar(self, parfilename, timfilename, engine='libstempo', \
            testpulsar=False):
        """
        Open a pulsar, given a parfile and a timfile

        @param parfilename: The name of the parfile to open
        @param timfilename: The name fo the timfile to open
        @param engine:      Which pulsar timing engine to use [libstempo]
        @param testpulsar:  If True, open the test pulsar (J1744, NANOGrav)
        """
        if testpulsar:
            # Write a temporary test pulsar file
            parfilename = tempfile.mktemp()
            timfilename = tempfile.mktemp()
            parfile = open(parfilename, 'w')
            timfile = open(timfilename, 'w')
            parfile.write(constants.J1744_parfile)
            timfile.write(constants.J1744_timfile)
            parfile.close()
            timfile.close()
        else:
            # Obtain the directory name of the timfile and relative path
            timfiletup = os.path.split(timfilename)
            dirname = timfiletup[0]
            timfilename = timfiletup[-1]
            parfilename = os.path.relpath(parfilename, dirname)
            savedir = os.getcwd()

            # Change directory to the base directory of the tim-file to deal with
            # INCLUDE statements in the tim-file
            if dirname != '':
                os.chdir(dirname)

        # Load the pulsar (and make the history available)
        # TODO: Also set the priors, logfile, loglevel, delete_prob, and mP0
        cell = "pspsr = qp.PSPulsar('"+parfilename+"', '"+timfilename+\
                "', backend='"+engine+"')"
        self.kernel.shell.run_cell(cell)
        cell = "psr = pspsr._psr"
        self.kernel.shell.run_cell(cell)
        cell = "history = []"
        self.kernel.shell.run_cell(cell)
        if engine == "pint":
            cell = "model = psr.model ; toas = psr.t"
            self.kernel.shell.run_cell(cell)

        try:
            pspsr = self.kernel.shell.ns_table['user_local']['pspsr']
            history = self.kernel.shell.ns_table['user_local']['history']
        except KeyError as err:
            pspsr = qp.PSPulsar(parfilename, timfilename, backend=engine)
            history = []

        if testpulsar:
            os.remove(parfilename)
            os.remove(timfilename)
        elif dirname != '':
            os.chdir(savedir)

        # Update the plk widget
        self.plkWidget.setPulsar(pspsr, history)

        # Communicating with the kernel goes as follows
        # self.kernel.shell.push({'foo': 43, 'print_process_id': print_process_id}, interactive=True)
        # print("Embedded, we have:", self.kernel.shell.ns_table['user_local']['foo'])

    def openBinaryPulsar(self, parfilename=None, perfilename=None, \
            testpulsar=False):
        """
        Open a pulsar, given a .bestprof file, and perhaps a par file

        @param parfilename: The name of the par/ephemeris file to open
        @param perfilename: The name of the .bestprof file to open
        @param testpulsar:  If True, open the test pulsar (J1756)
        """
        if testpulsar or perfilename is None:
            # Need to load the test pulsar
            tperfilename = tempfile.mktemp()
            tperfile = open(tperfilename, 'w')
            tperfile.write(constants.J1903PER)
            #tperfile.write(constants.J1756PER)
            tperfile.close()
            ms = True
        else:
            tperfilename = perfilename
            ms = False

        # Load the per-file
        cell = "bpsr = lo.orbitpulsar()"
        self.kernel.shell.run_cell(cell)
        cell = "bpsr.readPerFile('" + tperfilename +"', ms=" + str(ms) + ")"
        self.kernel.shell.run_cell(cell)

        if testpulsar or perfilename is None:
            os.remove(tperfilename)

        if testpulsar:
            tparfilename = tempfile.mktemp()
            tparfile = open(tparfilename, 'w')
            tparfile.write(constants.J1903EPH)
            #tperfile.write(constants.J1756PER)
            tparfile.close()
            cell = "bpsr.readParFile('" + tparfilename +"')"
            self.kernel.shell.run_cell(cell)
            os.remove(tparfilename)
        elif parfilename is not None:
            cell = "bpsr.readParFile('" + parfilename +"')"
            self.kernel.shell.run_cell(cell)

        bpsr = self.kernel.shell.ns_table['user_local']['bpsr']

        self.binaryWidget.setPulsar(bpsr)


    def keyPressEvent(self, event, **kwargs):
        """
        Handle a key-press event

        @param event:   event that is handled here
        """

        key = event.key()

        if key == QtCore.Qt.Key_Escape:
            self.close()
        elif key == QtCore.Qt.Key_Left:
            #print("Left pressed")
            pass
        else:
            #print("Other key")
            pass

        #print("QtipWindow: key press")
        super(QtipWindow, self).keyPressEvent(event, **kwargs)

    def mousePressEvent(self, event, **kwargs):
        """
        Handle a mouse-click event

        @param event:   event that is handled here
        """
        #print("QtipWindow: mouse click")
        super(QtipWindow, self).mousePressEvent(event, **kwargs)

    def preExecute(self):
        """
        Callback function that is run prior to execution of a cell
        """
        pass

    def postExecute(self):
        """
        Callback function that is run after execution of a code
        """
        pass

    def postRunCell(self):
        """
        Callback function that is run after execution of a cell (after
        post-execute)
        """
        # TODO: Do more than just update the plot, but also update _all_ the
        # widgets. Make a callback in plkWidget for that. QtipWindow might also
        # want to loop over some stuff.
        if self.whichWidget == 'plk':
            self.plkWidget.updatePlot()
        elif self.whichWidget == 'binary':
            self.binaryWidget.updatePlot()
        
def main():
    # The option parser
    usage = "usage: %prog [options]"
    parser = optparse.OptionParser(usage=usage)

    parser.add_option('-f', '--file', action='store', type='string', nargs=2, \
            default=(None, None), help="Provide a parfile and a timfile")

    parser.add_option('-p', '--periodfile', action='store', type='string', nargs=1, \
            default=(None, None), help="Provide a period file (per)")

    parser.add_option('-e', '--engine', action='store', type='string', nargs=1, \
            default='libstempo', \
            help="Pulsar timing engine: libstempo/pint/piccard")

    (options, args) = parser.parse_args()

    # Create the application
    app = QtGui.QApplication(sys.argv)

    # Create the window, and start the application
    qtipwin = QtipWindow(engine=options.engine, \
            parfile=options.file[0], timfile=options.file[1],\
            perfile=options.periodfile[0])
    qtipwin.raise_()        # Required on OSX to move the app to the foreground
    sys.exit(app.exec_())



if __name__ == '__main__':
    main()
