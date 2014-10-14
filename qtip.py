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
from optparse import OptionParser

# Numpy etc.
import numpy as np
import time
import matplotlib

import qtpulsar as qp
import constants

# Import libstempo and Piccard
#try:
#    import piccard as pic
#except ImportError:
#    pic is None

#try:
#    import libstempo as t2
#except ImportError:
#    t2 = None

from plk import *

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


# Parse the command-line options
parser = OptionParser()

# Add some options with add_option
#parser.add_option("-d", "--dataOnly",
#                  action="store_true", dest="dataOnly", default=False,
#                  help="Just include the data in h5 file [default false]") 

(options, args) = parser.parse_args()

if len(args) >= 2:
    partile = args[0]
    timfile = args[1]
else:
    parfile = None
    timfile = None


class PiccardWidget(QtGui.QWidget):
    """
    The Piccard main window (Not yet implemented)
    """
    def __init__(self, parent=None, **kwargs):
        super(PiccardWidget, self).__init__(parent, **kwargs)

        self.parent = parent

        self.initPiccard()

    def initPiccard(self):
        print("Hello, Piccard!")

class OpenSomethingWidget(QtGui.QWidget):
    """
    The open-something Widget. First shown in the main window, if it is not
    started with a command to open any file to begin with. This way, we don't
    have to show an empty graph
    """
    def __init__(self, parent=None, openFile=None, **kwargs):
        super(OpenSomethingWidget, self).__init__(parent, **kwargs)

        self.parent = parent
        self.openFileFn = openFile

        self.initOSWidget()

    def initOSWidget(self):
        """
        Initialise the widget with a button and a label
        """
        self.vbox = QtGui.QVBoxLayout()

        button = QtGui.QPushButton("Open a file...")
        button.clicked.connect(self.openFile)
        self.vbox.addWidget(button)

        self.setLayout(self.vbox)

    def openFile(self):
        """
        Display the open file dialog, and send the parent window the order to
        open the file
        """
        if not self.openFileFn is None:
            filename = QtGui.QFileDialog.getOpenFileName(self, 'Open file', '~/')
            self.openFileFn(filename)



class QtipWindow(QtGui.QMainWindow):
    """
    Main Qtip window

    Note, is the main window now, but the content will later be moved to a
    libstempo tab, as part of the Piccard suite
    """
    
    def __init__(self, parent=None):
        super(QtipWindow, self).__init__(parent)
        self.setWindowTitle('QtIpython interface to Piccard/libstempo')
        
        # Initialise basic gui elements
        self.initUI()

        # Start the embedded IPython kernel
        self.createIPythonKernel()

        # Create the display widgets
        self.createPlkWidget()
        self.createIPythonWidget()
        self.createOpenSomethingWidget()

        # Position the widgets
        self.initQtipLayout()
        self.setQtipLayout(whichWidget='opensomething', showIPython=True)

        # We are still in MAJOR testing mode, so open a test-pulsar right away
        # (delete this line when going into production)
        self.requestOpenPlk(testpulsar=True, engine='pint')

        self.show()

    def __del__(self):
        pass

    def onAbout(self):
        msg = """ A demo of using PyQt with matplotlib, libstempo, and IPython:
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
        self.openParTimAction = QtGui.QAction('&Open', self)        
        self.openParTimAction.setShortcut('Ctrl+O')
        self.openParTimAction.setStatusTip('Open par/tim')
        self.openParTimAction.triggered.connect(self.openParTim)

        self.exitAction = QtGui.QAction(QtGui.QIcon('exit.png'), '&Exit', self)        
        self.exitAction.setShortcut('Ctrl+Q')
        self.exitAction.setStatusTip('Exit application')
        self.exitAction.triggered.connect(self.close)

        self.toggleIPythonAction = QtGui.QAction('&IPython', self)        
        self.toggleIPythonAction.setShortcut('Ctrl+I')
        self.toggleIPythonAction.setStatusTip('Toggle IPython')
        self.toggleIPythonAction.triggered.connect(self.toggleIPython)

        self.aboutAction = QtGui.QAction('&About', self)        
        self.aboutAction.setShortcut('Ctrl+A')
        self.aboutAction.setStatusTip('About Qtip')
        self.aboutAction.triggered.connect(self.onAbout)

        self.statusBar()
        
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
        self.fileMenu.addAction(self.exitAction)
        self.viewMenu = self.menubar.addMenu('&View')
        self.viewMenu.addAction(self.toggleIPythonAction)
        self.helpMenu = self.menubar.addMenu('&Help')
        self.helpMenu.addAction(self.aboutAction)

        # What is the status quo of the user interface?
        self.showIPython = False
        self.whichWidget = 'None'
        self.prevShowIPython = False
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
        cell = "import numpy as np, matplotlib.pyplot as plt, qtpulsar as qp"
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
        self.consoleWidget.setMinimumSize(600, 550)
        self.consoleWidget.banner = QtipBanner
        self.consoleWidget.kernel_manager = self.kernelManager
        self.consoleWidget.kernel_client = self.kernelClient
        self.consoleWidget.exit_requested.connect(self.toggleIPython)
        self.consoleWidget.set_default_style(colors='linux')
        self.consoleWidget.hide()


    def createOpenSomethingWidget(self):
        """
        Create the OpenSomething widget. Do not add it to the layout yet

        TODO:   probably should use a signal to implement this call instead of
                openParTim
        """
        self.openSomethingWidget = OpenSomethingWidget(parent=self.mainFrame, \
                openFile=self.requestOpenPlk)
        self.openSomethingWidget.hide()

    def createPlkWidget(self):
        """
        Create the Plk widget
        """
        self.plkWidget = PlkWidget(parent=self.mainFrame)
        self.plkWidget.hide()


    def toggleIPython(self):
        """
        Toggle the IPython widget on or off
        """
        self.showIPython = not self.showIPython

        self.setQtipLayout()

    def initQtipLayout(self):
        """
        Initialise the Qtip layout
        """
        self.mainFrame.setMinimumSize(1350, 550)
        self.hbox.addWidget(self.openSomethingWidget)
        self.hbox.addWidget(self.plkWidget)
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

        if self.showIPython:
            self.mainFrame.setMinimumSize(1300, 550)
            self.consoleWidget.show()
        else:
            self.mainFrame.setMinimumSize(650, 550)

        if self.whichWidget.lower() == 'plk' and not self.showIPython:
            #self.plkWidget.setFocus()
            self.plkWidget.setFocusToCanvas()
        #elif self.showIPython:
        #    self.consoleWidget.setFocus()

    def setQtipLayout(self, whichWidget=None, showIPython=None):
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
        self.prevShowIPython = self.showIPython

        self.mainFrame.setLayout(self.hbox)
        #print("whichWidget = {0},  showIPython = {1}".format(self.whichWidget, self.showIPython))
        self.mainFrame.show()


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
        self.openPulsar(parfilename, timfilename, engine=engine, \
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


    def openParTim(self, filename=None, engine='libstempo'):
        """
        Open a par-file and a tim-file
        """
        print("openParTim called with {0}".format(filename))

        # Ask the user for a par and tim file, and open these with libstempo
        if isinstance(filename, str):
            parfilename = filename
        else:
            parfilename = QtGui.QFileDialog.getOpenFileName(self, 'Open par-file', '~/')

        timfilename = QtGui.QFileDialog.getOpenFileName(self, 'Open tim-file', '~/')

        # Load the pulsar
        self.openPulsar(parfilename, timfilename, engine=engine)

    def openPulsar(self, parfilename, timfilename, engine='libstempo',
            testpulsar=False):
        """
        Open a pulsar, given a parfile and a timfile

        @param parfilename: The name of the parfile to open
        @param timfilename: The name fo the timfile to open
        @param engine:      Which pulsar timing engine to use [libstempo]
        @param testpulsar:  If True, open the test pulsar (J1744, NANOGrav)
        """
        if engine=='pint':
            trypint = True
        else:
            trypint = False

        engine, pclass = qp.get_engine(trypint=trypint)

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
                os.chdir(dirname)

                # Load the pulsar
                cell = "psr = qp."+pclass+"('"+parfilename+"', '"+timfilename+"')"
                self.kernel.shell.run_cell(cell)
                psr = self.kernel.shell.ns_table['user_local']['psr']

                # Change directory back to where we were
                os.chdir(savedir)
            else:
                cell = "psr = qp."+pclass+"(testpulsar=True)"
                self.kernel.shell.run_cell(cell)
                psr = self.kernel.shell.ns_table['user_local']['psr']
        elif engine == 'pint':
            cell = "psr = qp."+pclass+"(testpulsar=True)"
            self.kernel.shell.run_cell(cell)
            psr = self.kernel.shell.ns_table['user_local']['psr']
        else:
            raise NotImplemented("Only works with PINT/libstempo")

        # Update the plk widget
        self.plkWidget.setPulsar(psr)

        # Communicating with the kernel goes as follows
        # self.kernel.shell.push({'foo': 43, 'print_process_id': print_process_id}, interactive=True)
        # print("Embedded, we have:", self.kernel.shell.ns_table['user_local']['foo'])


    def keyPressEvent(self, event, **kwargs):
        """
        Handle a key-press event

        @param event:   event that is handled here
        """

        key = event.key()

        if key == QtCore.Qt.Key_Escape:
            self.close()
        elif key == QtCore.Qt.Key_Left:
            print("Left pressed")
        else:
            print("Other key")

        print("QtipWindow: key press")
        super(QtipWindow, self).keyPressEvent(event, **kwargs)

    def mousePressEvent(self, event, **kwargs):
        """
        Handle a mouse-click event

        @param event:   event that is handled here
        """
        print("QtipWindow: mouse click")
        super(QtipWindow, self).mousePressEvent(event, **kwargs)

        
def main():
    app = QtGui.QApplication(sys.argv)
    qtipwin = QtipWindow()
    qtipwin.raise_()        # Required on OSX to move the app to the foreground
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
