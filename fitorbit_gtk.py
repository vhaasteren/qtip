#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: tabstop=4:softtabstop=4:shiftwidth=4:expandtab
"""
fitorbit: Qt port of fitorbit

"""

from __future__ import print_function
from __future__ import division


#import gobject
#import gtk
from PyQt4 import QtGui, QtCore

import os, sys
import time

from optparse import OptionParser

from matplotlib.figure import Figure
from scipy.optimize import leastsq
import math 
import slalib
from pyslalib import *         # For Degrees/Arcmin/Arcsec to Radians, and Hour/Min/Sec to Radians

import numpy as np

import matplotlib
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
#matplotlib.use('GtkAgg')
#from matplotlib.backends.backend_gtkagg import FigureCanvasGTKAgg as FigureCanvas
#from matplotlib.backends.backend_gtk import NavigationToolbar2GTK as NavigationToolbar
from matplotlib.ticker import ScalarFormatter

import parfile
from utils import eccentric_anomaly

DEG2RAD    = float('1.7453292519943295769236907684886127134428718885417e-2')
RAD2DEG    = float('57.295779513082320876798154814105170332405472466564')
C           = float('2.99792458e8')

full_usage = """
usage : fitorbit.py [options] File

  [-h, --help]        : Display this help
  [-f, --freq]        : Convert File from frequency
  [-m, --ms]          : Convert File from period in ms

  File is a list of 
  MJDs  Period/Freq  Unc

"""

usage = "usage: %prog [options]"  


PARAMS = ['RA', 'DEC', 'P0', 'P1', 'PEPOCH', 'PB', 'ECC', 'A1', 'T0', 'OM']

class Param:
    def __init__(self, is_string=False):
        self.val = 0.0
        if is_string:
            self.val = "00:00:00.0"
        self.fit = 0


# Quit Function
def quit(action, self):
    gtk.main_quit();
    return False

def activate_action(action,self):
    print 'Action "%s" activated' % action.get_name()


def readfile(filename, flgfreq, flgms):
    """
    Open a file 'filename and
    Return MJD and Period for N-pts
    """
    mjds, ps = np.loadtxt(filename, usecols=(0,1), unpack=True)

    # Frequency given in data
    if flgfreq:
      ps=1/ps
    # Period in ms given in data  
    if flgms:
      ps=ps*1.
    else:
      ps=ps*1000.
    return mjds, ps

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


# Declarations for MENU
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



#class Manager(gtk.Window):
class Manager(QtGui.QMainWindow):

    def __init__(self, filename, parent=None):

        super(Manager, self).__init__(parent)

        self.mjds, self.periods = readfile(filename, opts.freq, opts.ms)

        # Variables Init
        self.init_param_file()


        # Main window (Widget)
        self.mainFrame = QtGui.QWidget()
        self.setCentralWidget(self.mainFrame)


        # Build Main window
        #gtk.Window.__init__(self)
        #try:
        #    self.set_screen(parent.get_screen())
        #except AttributeError:
        #    self.connect('destroy', quit, self)
        #self.set_title("Fit Orbit")
        #self.set_size_request(800,600)
        #self.set_border_width(0)
        self.resize(800, 600)
        self.mainFrame.resize(800, 600)
        self.move(200, 100)


        # Build main menu
        self.actions = gtk.ActionGroup("Actions")
        self.actions.add_actions(entries,self)
        ui = gtk.UIManager()
        ui.insert_action_group(self.actions, 0)
        self.add_accel_group(ui.get_accel_group())
        try:
            mergeid = ui.add_ui_from_string(ui_info)
        except gobject.GError, msg:
            print "building menus failed: %s" % msg

        # Add box for main menu
        box1 = gtk.VBox(False, 0)
        self.add(box1)
        box1.pack_start(ui.get_widget("/MenuBar"), False, False, 0)

        self.box12 = gtk.HBox(False, 0)
        box1.pack_start(self.box12, True, True, 0)

        # Add box for options menu
        self.box_opt = gtk.HBox(False, 0)
        self.draw_options()
        self.box12.pack_start(self.box_opt, False, False, 0)

        # Add box for parameters menu
        self.box_param = gtk.VBox(False, 0)
        self.draw_param()
        self.box12.pack_start(self.box_param, True, True, 0)

        # Add graphic box and display Label
        self.xlabel="MJD"
        self.ylabel="Period (ms)"
        self.fig = Figure(facecolor='white')
        self.ax1 = self.fig.add_subplot(1,1,1)

        ###########
        left, width = 0.1, 0.8
        rect1 = [left, 0.1, width, 0.7]
        rect2 = [left, 0.8, width, 0.1]


        self.ax1.set_xlabel(self.xlabel)
        self.ax1.set_ylabel(self.ylabel)
        self.ax1.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        self.ax1.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        self.canvas = FigureCanvas(self.fig)  
        self.box_param.pack_start(self.canvas, True, True, 0)

        # Add Toolbar box
        toolbar = NavigationToolbar(self.canvas, self)
        self.box_param.pack_start(toolbar, False, False)

        # plot
        self.ax1.plot(self.mjds,self.periods,'r+',ms=9)

        self.show_all()

    def init_param_file(self):
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


    def read_param_file(self):
        self.param.read(self.param_filename)

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

    def write_param_file(self):
        for PARAM in PARAMS:
            self.param.set_param(PARAM, self.p2f[PARAM].val)

    def plot_model(self, widget=None):

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


    def fit_model(self, widget=None):
        """
        Function to perform the fit of selected parameters to the values
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



    def read_param_file_select(self, widget):

        # Setup the dialog box
        dialog = gtk.FileChooserDialog("Open Parfile",None,gtk.FILE_CHOOSER_ACTION_OPEN, (gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL, gtk.STOCK_OK, gtk.RESPONSE_OK))
        dialog.set_default_response(gtk.RESPONSE_OK)
        response = dialog.run()
        if response == gtk.RESPONSE_OK:
            self.param_filename = dialog.get_filename()
            self.read_param_file()
        elif response == gtk.RESPONSE_CANCEL:
            print 'Closed'
        dialog.destroy()  

        for i,val in enumerate(self.p2f):
            self.local_entry[i].set_text(str(self.p2f[self.label[i]].val))

        self.show_all()

    def write_param_file_select(self, widget):

        # Setup the dialog box
        dialog = gtk.FileChooserDialog("Save Parfile",None,gtk.FILE_CHOOSER_ACTION_SAVE, (gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL, gtk.STOCK_OK, gtk.RESPONSE_OK))
        dialog.set_default_response(gtk.RESPONSE_OK)
        response = dialog.run()

        #
        if response == gtk.RESPONSE_OK:
            new_filename = dialog.get_filename()
            self.p2f['P1'].val = self.param.P1*1e-15

            # Update the parameters in the Parfile Class
            self.write_param_file()
            # Write the file 
            self.param.write(new_filename)
            self.p2f['P1'].val = self.param.P1/1e-15
        elif response == gtk.RESPONSE_CANCEL:
            print 'Closed'

        dialog.destroy()  

    def key_press_menu(self, event):
        """
        """

        if event.key=='x':
            self.fit_model()


    def draw_options(self):

        #self.frame1 = gtk.Frame()
        #self.frame1.set_shadow_type(gtk.SHADOW_IN)
        #self.frame1.set_size_request(60, 60)

        self.box_buttons = gtk.VBox(False, 0)
        self.box_opt.pack_start(self.box_buttons, False, False, 0)

        # Button "Load Parfile"
        button_load_par = gtk.Button("Load ParFile")
        self.box_buttons.pack_start(button_load_par, False, False, 0)
        button_load_par.connect("clicked", self.read_param_file_select)

        # Button "Plot Model"
        button_plot_model = gtk.Button("Plot Model")
        self.box_buttons.pack_start(button_plot_model, False, False, 0)
        button_plot_model.connect("clicked", self.plot_model)

        # Button "Fit"
        button_fit = gtk.Button("Fit Model")
        self.box_buttons.pack_start(button_fit, False, False, 0)
        button_fit.connect("clicked", self.fit_model)

        # Button "Save"
        button_save = gtk.Button("Save Parfile")
        self.box_buttons.pack_start(button_save, False, False, 0)
        button_save.connect("clicked", self.write_param_file_select)


    # Which param to held fixed
    def set_fit(self, widget, data=None):
        print "Parameter %s was toggled %s" % (data, ("OFF", "ON")[widget.get_active()])

        status = ("OFF", "ON")[widget.get_active()]

        if status=='ON':
            self.fit_flag[self.label.index(data)]=1
        elif status=='OFF':
            self.fit_flag[self.label.index(data)]=0


    def draw_param(self):
        """
        Draw parameters menu
        """
        table = gtk.Table(5,6, False)
        table.set_row_spacings(4)
        table.set_col_spacings(4)
        self.box_param.pack_start(table, False, True, 0)

        self.local_entry=[]

        # RA
        i=0
        check_button = gtk.CheckButton()
        check_button.connect('toggled', self.set_fit, "RA")
        table.attach(check_button, 0, 1, 0, 1, gtk.SHRINK)
        label = gtk.Label("RA")
        table.attach(label, 1, 2, 0, 1, gtk.SHRINK)
        self.local_entry.append(gtk.Entry())

        self.local_entry[i].set_text(str(self.p2f['RA'].val))
        table.attach(self.local_entry[i], 2, 3, 0, 1, gtk.EXPAND|gtk.FILL)

        # DEC
        i+=1
        check_button = gtk.CheckButton()
        check_button.connect('toggled', self.set_fit, "DEC")
        table.attach(check_button, 0, 1, 1, 2, gtk.SHRINK)
        label = gtk.Label("DEC")
        table.attach(label, 1, 2, 1, 2, gtk.SHRINK)
        self.local_entry.append(gtk.Entry())
        self.local_entry[i].set_text(str(self.p2f['DEC'].val))
        table.attach(self.local_entry[i], 2, 3, 1, 2, gtk.EXPAND|gtk.FILL)

        # P0
        i+=1
        check_button = gtk.CheckButton()
        check_button.connect('toggled', self.set_fit, "P0")
        table.attach(check_button, 0, 1, 2, 3, gtk.SHRINK)
        label = gtk.Label("P (s)")
        table.attach(label, 1, 2, 2, 3, gtk.SHRINK)
        self.local_entry.append(gtk.Entry())
        self.local_entry[i].set_text(str(self.p2f['P0'].val))
        table.attach(self.local_entry[i], 2, 3, 2, 3, gtk.EXPAND|gtk.FILL)

        # P1
        i+=1
        check_button = gtk.CheckButton()
        check_button.connect('toggled', self.set_fit, "P1")
        table.attach(check_button, 0, 1, 3, 4, gtk.SHRINK)
        label = gtk.Label("Pdot(e-15)")
        table.attach(label, 1, 2, 3, 4, gtk.SHRINK)
        self.local_entry.append(gtk.Entry())
        self.local_entry[i].set_text(str(self.p2f['P1'].val))
        table.attach(self.local_entry[i], 2, 3, 3, 4, gtk.EXPAND|gtk.FILL)

        # Pepoch
        i+=1
        check_button = gtk.CheckButton()
        check_button.connect('toggled', self.set_fit, "PEPOCH")
        table.attach(check_button, 0, 1, 4, 5, gtk.SHRINK)
        label = gtk.Label("Pepoch")
        table.attach(label, 1, 2, 4, 5, gtk.SHRINK)
        self.local_entry.append(gtk.Entry())
        self.local_entry[i].set_text(str(self.p2f['PEPOCH'].val))
        table.attach(self.local_entry[i], 2, 3, 4, 5, gtk.EXPAND|gtk.FILL)


        # Second colum of parameters

        i+=1
        check_button = gtk.CheckButton()
        check_button.connect('toggled', self.set_fit, "PB")
        table.attach(check_button, 3, 4, 0, 1, gtk.SHRINK)
        label = gtk.Label("Porb")
        table.attach(label, 4, 5, 0, 1, gtk.SHRINK)
        self.local_entry.append(gtk.Entry())
        self.local_entry[i].set_text(str(self.p2f['PB'].val))
        table.attach(self.local_entry[i], 5, 6, 0, 1, gtk.EXPAND|gtk.FILL)

        i+=1
        check_button = gtk.CheckButton()
        check_button.connect('toggled', self.set_fit, "ECC")
        table.attach(check_button, 3, 4, 1, 2, gtk.SHRINK)
        label = gtk.Label("Ecc")
        table.attach(label, 4, 5, 1, 2, gtk.SHRINK)
        self.local_entry.append(gtk.Entry())
        self.local_entry[i].set_text(str(self.p2f['ECC'].val))
        table.attach(self.local_entry[i], 5, 6, 1, 2, gtk.EXPAND|gtk.FILL)

        i+=1
        check_button = gtk.CheckButton()
        check_button.connect('toggled', self.set_fit, "A1")
        table.attach(check_button, 3, 4, 2, 3, gtk.SHRINK)
        label = gtk.Label("A1")
        table.attach(label, 4, 5, 2, 3, gtk.SHRINK)
        self.local_entry.append(gtk.Entry())
        self.local_entry[i].set_text(str(self.p2f['A1'].val))
        table.attach(self.local_entry[i], 5, 6, 2, 3, gtk.EXPAND|gtk.FILL)

        i+=1
        check_button = gtk.CheckButton()
        check_button.connect('toggled', self.set_fit, "T0")
        table.attach(check_button, 3, 4, 3, 4, gtk.SHRINK)
        label = gtk.Label("T0")
        table.attach(label, 4, 5, 3, 4, gtk.SHRINK)
        self.local_entry.append(gtk.Entry())
        self.local_entry[i].set_text(str(self.p2f['T0'].val))
        table.attach(self.local_entry[i], 5, 6, 3, 4, gtk.EXPAND|gtk.FILL)

        # OM
        i+=1
        check_button = gtk.CheckButton()
        check_button.connect('toggled', self.set_fit, "OM")
        table.attach(check_button, 3, 4, 4, 5, gtk.SHRINK)
        label = gtk.Label("Om")
        table.attach(label, 4, 5, 4, 5, gtk.SHRINK)
        self.local_entry.append(gtk.Entry())
        self.local_entry[i].set_text(str(self.p2f['OM'].val))
        table.attach(self.local_entry[i], 5, 6, 4, 5, gtk.EXPAND|gtk.FILL)


if __name__ == '__main__':

    usage = "usage: %prog [options] -f period.dat"  

    parser = OptionParser(usage)
    parser.add_option("-c", "--convert_f", action="store_true", dest="freq", default=False, help="Use frequency")
    parser.add_option("-m", "--ms", action="store_true", dest="ms", default=False, help="Use period in ms")
    parser.add_option("-f", "--file", type="string", dest="filename", help="Input file")

    (opts, args) = parser.parse_args()

    if len(args)==0:
        print full_usage
        sys.exit(0)

    if gtk.pygtk_version <(2,3,90):
        print "PyGtk 2.3.90 or later is required"
        raise SystemExit

    Manager(sys.argv[1])
    gtk.main()

