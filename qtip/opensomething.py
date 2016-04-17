#!/usr/bin/python
# -*- coding: utf-8 -*-
# vim: tabstop=4:softtabstop=4:shiftwidth=4:expandtab

"""
OpenSomething: Qt widget to open a file


"""

from __future__ import print_function
from __future__ import division
import os, sys

# Importing all the stuff for the IPython console widget
from PyQt4 import QtGui, QtCore

# Importing all the stuff for the matplotlib widget
import matplotlib
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas

try:
    from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
except ImportError:
    from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar

from matplotlib.figure import Figure

# Numpy etc.
import numpy as np


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

