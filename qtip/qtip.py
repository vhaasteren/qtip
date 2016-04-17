#!/usr/bin/python
# -*- coding: utf-8 -*-
# vim: tabstop=4:softtabstop=4:shiftwidth=4:expandtab
"""
qtip: Qt interactive interface for PTA data analysis tools

"""


from __future__ import print_function
from __future__ import division
import os, sys

# Importing all the stuff for the Jupyter console widget
from qtconsole.rich_jupyter_widget import RichJupyterWidget
from qtconsole.inprocess import QtInProcessKernelManager
from qtconsole.qt import QtCore, QtGui

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


class QtipWindow(QtGui.QMainWindow):
    """Main Qtip window class"""
    
    def __init__(self, parent=None, engine='pint', \
            parfile=None, timfile=None, **kwargs):
        """
        Initialize the main window

        :param parent:
            The parent window that embeds the Qtip window

        :param engine:
            The preferred timing package to use ('pint'/'libstempo') ['pint']

        :param parfile:
            Par-file top open on startup

        :param timfile:
            Tim-file top open on startup

        """

        super(QtipWindow, self).__init__(parent)
        self.setWindowTitle('Jupyter interface for pulsar timing')

        # Initialise basic gui elements
        self.initUI()

        # Start the embedded Jupyter kernel
        self.createJupyterKernel()

        # Create the display widgets
        self.createPlkWidget()
        self.createJupyterWidget()
        self.createOpenSomethingWidget()

        # Position the widgets
        self.initQtipLayout()

        # Initialize the main widget (the plk emulator)
        self.setQtipLayout(whichWidget='plk',
                showJupyter=False, firsttime=True)

        # The preferred engine to use (PINT)
        self.pref_engine = engine

        # We are still in MAJOR testing mode, so open a test-pulsar right away
        # if no par/tim file is given
        if parfile is None or timfile is None:
            testpulsar = True
        else:
            testpulsar = False

        # Open plk as the main widget
        self.requestOpenPlk(testpulsar=testpulsar, parfilename=parfile, \
                timfilename=timfile, engine=self.pref_engine)

        self.show()

    def __del__(self):
        pass

    def onAbout(self):
        """Show an about box"""

        msg = constants.QtipBanner
        QtGui.QMessageBox.about(self, "About Qtip", msg.strip())

    def initUI(self):
        """Initialise the user-interface elements"""

        # Create the main-frame widget, and the layout
        self.mainFrame = QtGui.QWidget()
        self.setCentralWidget(self.mainFrame)
        self.hbox = QtGui.QHBoxLayout()     # HBox contains all widgets

        # Menu item: open par/tim files
        self.openParTimAction = QtGui.QAction('&Open par/tim', self)        
        self.openParTimAction.setShortcut('Ctrl+O')
        self.openParTimAction.setStatusTip('Open par/tim')
        self.openParTimAction.triggered.connect(self.openParTim)

        # Menu item: exit Qtip
        self.exitAction = QtGui.QAction(QtGui.QIcon('exit.png'), '&Exit', self)        
        self.exitAction.setShortcut('Ctrl+Q')
        self.exitAction.setStatusTip('Exit application')
        self.exitAction.triggered.connect(self.close)

        # Previously, it was possible to switch out the 'plk' widget for another
        # main widget (the binary pulsar one). That one has been stripped out
        # now, so for now it makes no sense to 'toggle' on or off the plk
        # widget. However, the option is still there for now...
        self.togglePlkAction = QtGui.QAction('&Plk', self)        
        self.togglePlkAction.setShortcut('Ctrl+P')
        self.togglePlkAction.setStatusTip('Toggle plk widget')
        self.togglePlkAction.triggered.connect(self.togglePlk)

        # Menu item: toggle the Jupyter window
        self.toggleJupyterAction = QtGui.QAction('&Jupyter', self)        
        self.toggleJupyterAction.setShortcut('Ctrl+J')
        self.toggleJupyterAction.setStatusTip('Toggle Jupyter')
        self.toggleJupyterAction.triggered.connect(self.toggleJupyter)

        # Menu item: about Qtip
        self.aboutAction = QtGui.QAction('&About', self)        
        self.aboutAction.setShortcut('Ctrl+A')
        self.aboutAction.setStatusTip('About Qtip')
        self.aboutAction.triggered.connect(self.onAbout)

        # The status bar
        self.theStatusBar = QtGui.QStatusBar()
        #self.statusBar()
        self.setStatusBar(self.theStatusBar)

        # A label that shows what engine is being used (hardcoded: PINT)
        self.engine_label = QtGui.QLabel("PINT")
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

        # Create the menu bar, and link the action items
        self.menubar = self.menuBar()
        self.fileMenu = self.menubar.addMenu('&File')
        self.fileMenu.addAction(self.openParTimAction)
        self.fileMenu.addAction(self.exitAction)
        self.viewMenu = self.menubar.addMenu('&View')
        self.viewMenu.addAction(self.togglePlkAction)
        self.viewMenu.addAction(self.toggleJupyterAction)
        self.helpMenu = self.menubar.addMenu('&Help')
        self.helpMenu.addAction(self.aboutAction)

        # What is the status quo of the user interface?
        self.showJupyter = False
        self.whichWidget = 'None'
        self.prevShowJupyter = None
        self.prevWhichWidget = 'None'

    def createJupyterKernel(self):
        """Create and start the embedded Jupyter Kernel"""

        # Create an in-process kernel
        self.kernelManager = QtInProcessKernelManager()
        self.kernelManager.start_kernel()
        self.kernel = self.kernelManager.kernel

        # Launch the kernel
        self.kernelClient = self.kernelManager.client()
        self.kernelClient.start_channels()

        # Allow inline matplotlib figures
        self.kernel.shell.enable_matplotlib(gui='inline')

        # Load the necessary packages in the embedded kernel
        # TODO: show this line in a cell of it's own
        cell = "import numpy as np, matplotlib.pyplot as plt, qtpulsar as qp"
        self.kernel.shell.run_cell(cell, store_history=False)

        # Set the in-kernel matplotlib color scheme to black.
        self.setMplColorScheme('black')     # Outside as well (do we need this?)
        self.kernel.shell.run_cell(constants.matplotlib_rc_cell_black,
                store_history=False)

    def createJupyterWidget(self):
        """Create the Jupyter widget"""

        self.consoleWidget = RichJupyterWidget()
        #self.consoleWidget.setMinimumSize(600, 550)

        # Show the banner
        self.consoleWidget.banner = constants.QtipBanner
        self.consoleWidget.kernel_manager = self.kernelManager

        # Couple the client
        self.consoleWidget.kernel_client = self.kernelClient
        self.consoleWidget.exit_requested.connect(self.toggleJupyter)
        self.consoleWidget.set_default_style(colors='linux')
        self.consoleWidget.hide()

        # Register a call-back function for the Jupyter shell. This one is
        # executed insite the child-kernel.
        #self.kernel.shell.register_post_execute(self.postExecute)
        #
        # In Jupyter >= 2, we can use the event register
        # Events: post_run_cell, pre_run_cell, etc...`
        self.kernel.shell.events.register('pre_execute', self.preExecute)
        self.kernel.shell.events.register('post_execute', self.postExecute)
        self.kernel.shell.events.register('post_run_cell', self.postRunCell)


    def createOpenSomethingWidget(self):
        """Create the OpenSomething widget. Do not add it to the layout yet

        TODO:   This widget should become the first main widget to see? At the
                moment, we're avoiding it for the sake of testing purposes
        """

        self.openSomethingWidget = OpenSomethingWidget(parent=self.mainFrame, \
                openFile=self.requestOpenPlk)
        self.openSomethingWidget.hide()

    def createPlkWidget(self):
        """Create the Plk widget"""

        self.plkWidget = PlkWidget(parent=self.mainFrame)
        self.plkWidget.hide()

    def toggleJupyter(self):
        """Toggle the Jupyter widget on or off"""

        self.setQtipLayout(showJupyter = not self.showJupyter)

    def togglePlk(self):
        """Toggle the plk widget on or off"""

        self.setQtipLayout(whichWidget='plk')

    def initQtipLayout(self):
        """Initialise the Qtip layout"""

        # If other 'main' widgets exist, they can be added here
        self.hbox.addWidget(self.openSomethingWidget)
        self.hbox.addWidget(self.plkWidget)

        self.hbox.addStretch(1)
        self.hbox.addWidget(self.consoleWidget)
        self.mainFrame.setLayout(self.hbox)

    def hideAllWidgets(self):
        """Hide all widgets of the mainFrame"""

        # Remove all widgets from the main window
        # No, hiding seems to work better
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
        self.consoleWidget.hide()

    def showVisibleWidgets(self):
        """Show the correct widgets in the mainFrame"""

        # Add the widgets we need
        if self.whichWidget.lower() == 'opensomething':
            self.openSomethingWidget.show()
        elif self.whichWidget.lower() == 'plk':
            self.plkWidget.show()
        # Other widgets can be added here

        if self.showJupyter:
            self.consoleWidget.show()
        else:
            pass

        # Request focus back to the main widget
        if self.whichWidget.lower() == 'plk' and not self.showJupyter:
            self.plkWidget.setFocusToCanvas()
        # Do it for other main widgets, if they exist
        #elif self.whichWidget.lower() == 'binary' and not self.showJupyter:
        #    self.binaryWidget.setFocusToCanvas()

        # Do we immediately get focus to the Jupyter console?
        #elif self.showJupyter:
        #    self.consoleWidget.setFocus()

    def setQtipLayout(self, whichWidget=None, showJupyter=None, firsttime=False):
        """Given the current main widget, hide all the other widgets
        
        :param whichWidget:
            Which main widget we are showing right now

        :param showJupyter:
            Whether to show the Jupyter console

        :param firsttime:
            Whether or not this is the first time setting the layout. If so,
            resize to proper dimensions.
            TODO: How to do this more elegantly?
        """

        if not whichWidget is None:
            self.whichWidget = whichWidget
        if not showJupyter is None:
            self.showJupyter = showJupyter

        # After hiding the widgets, wait 0 milliseonds before showing them again
        # (what a dirty hack, ugh!)
        self.hideAllWidgets()
        QtCore.QTimer.singleShot(0, self.showVisibleWidgets)

        self.prevWhichWidget = self.whichWidget

        if self.showJupyter != self.prevShowJupyter:
            # Jupyter has been toggled
            self.prevShowJupyter = self.showJupyter
            if self.showJupyter:
                self.resize(1350, 550)
                self.mainFrame.resize(1350, 550)
            else:
                self.resize(650, 550)
                self.mainFrame.resize(650, 550)

        # TODO: How to do this more elegantly?
        if firsttime:
            # Set position slightly more to the left of the screen, so we can
            # still open Jupyter
            self.move(50, 100)

        self.mainFrame.setLayout(self.hbox)
        self.mainFrame.show()

    def requestOpenPlk(self, parfilename=None, timfilename=None, \
            testpulsar=False, engine='pint'):
        """Request to open a file in the plk widget

        :param parfilename:
            The parfile to open. If none, ask the user
        :param timfilename:
            The timfile to open. If none, ask the user
        """

        self.setQtipLayout(whichWidget='plk', showJupyter=self.showJupyter)

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

        :param scheme: 'black'/'white', the color scheme
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
        """Open a par-file and a tim-file"""

        # Ask the user for a par and tim file, and open these with libstempo/pint
        parfilename = QtGui.QFileDialog.getOpenFileName(self, 'Open par-file', '~/')
        timfilename = QtGui.QFileDialog.getOpenFileName(self, 'Open tim-file', '~/')

        # Load the pulsar
        self.openPlkPulsar(parfilename, timfilename, engine=self.pref_engine)

    def openPlkPulsar(self, parfilename, timfilename, engine='pint', \
            testpulsar=False):
        """Open a pulsar, given a parfile and a timfile

        :param parfilename: The name of the parfile to open
        :param timfilename: The name fo the timfile to open
        :param engine:      Which pulsar timing engine to use [pint]
        :param testpulsar:  If True, open the test pulsar (J1744, NANOGrav)
        """

        if engine=='pint':
            trypint = True
        else:
            trypint = False

        engine, pclass = qp.get_engine(trypint=trypint)

        # This all is a bit ugly...
        if engine == 'libstempo':
            if not testpulsar:
                # Obtain the directory name of the timfile, and change to it
                timfiletup = os.path.split(timfilename)
                dirname = timfiletup[0]
                reltimfile = timfiletup[-1]
                relparfile = os.path.relpath(parfilename, dirname)
                savedir = os.getcwd()

                # Change directory to the base directory of the tim-file to deal with
                # INCLUDE statements in the tim-file
                if dirname != '':
                    os.chdir(dirname)

                # Load the pulsar
                cell = "psr = qp."+pclass+"('"+relparfile+"', '"+reltimfile+"')"
                self.kernel.shell.run_cell(cell)
                psr = self.kernel.shell.ns_table['user_local']['psr']

                # Change directory back to where we were
                if dirname != '':
                    os.chdir(savedir)
            else:
                cell = "psr = qp."+pclass+"(testpulsar=True)"
                self.kernel.shell.run_cell(cell)
                psr = self.kernel.shell.ns_table['user_local']['psr']
        elif engine == 'pint':
            if not testpulsar:
                psr = qp.PPulsar(parfilename, timfilename)
                cell = "psr = qp."+pclass+"('"+parfilename+"', '"+timfilename+"')"
            else:
                psr = qp.PPulsar(testpulsar=True)
                cell = "psr = qp."+pclass+"(testpulsar=True)"
            self.kernel.shell.run_cell(cell)
            psr = self.kernel.shell.ns_table['user_local']['psr']
        else:
            print("Engine = ", engine)
            raise NotImplemented("Only works with PINT/libstempo")

        # Update the plk widget
        self.plkWidget.setPulsar(psr)

        # Communicating with the kernel goes as follows
        # self.kernel.shell.push({'foo': 43, 'print_process_id': print_process_id}, interactive=True)
        # print("Embedded, we have:", self.kernel.shell.ns_table['user_local']['foo'])


    def keyPressEvent(self, event, **kwargs):
        """Handle a key-press event

        :param event:   event that is handled here
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
        """Handle a mouse-click event

        :param event:   event that is handled here
        """

        super(QtipWindow, self).mousePressEvent(event, **kwargs)

    def preExecute(self):
        """Callback function that is run prior to execution of a cell"""
        pass

    def postExecute(self):
        """Callback function that is run after execution of a code"""
        pass

    def postRunCell(self):
        """Callback function that is run after execution of a cell (after
        post-execute)
        """

        # TODO: Do more than just update the plot, but also update _all_ the
        # widgets. Make a callback in plkWidget for that. QtipWindow might also
        # want to loop over some stuff.
        if self.whichWidget == 'plk':
            #self.plkWidget.updatePlot()
            pass
        
def main():
    # The option parser
    usage = "usage: %prog [options]"
    parser = optparse.OptionParser(usage=usage)

    parser.add_option('-f', '--file', action='store', type='string', nargs=2, \
            default=(None, None), help="Provide a parfile and a timfile")

    parser.add_option('-p', '--periodfile', action='store', type='string', nargs=1, \
            default=(None, None), help="Provide a period file (per)")

    parser.add_option('-e', '--engine', action='store', type='string', nargs=1, \
            default='pint', \
            help="Pulsar timing engine: libstempo/pint")

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
