qtip
====

Project website: http://vhaasteren.github.io/qtip

Description
===========

Qt Interface for Pulsar timing

This is a graphical interface for Pulsar Timing packages. It is currently under
construction, but it is planned to be compatible with:

 * [Tempo2](http://tempo2.sourceforge.net)
 * [libstempo](https://github.com/vallis/mc3pta/tree/master/stempo)
 * [PINT](https://github.com/NANOGrav/PINT/)
 * [Piccard](https://github.com/vhaasteren/piccard)
 * ...

It works with an embedded IPython kernel. That's where all the calculations are
performed.

At the moment, not all the extended functions of Plk are implemented.

Requirements:
=============

 * [numpy](http://numpy.scipy.org)
 * [scipy](http://numpy.scipy.org)
 * [matplotlib](http://matplotlib.org), for plotting only
 * [tempo2](http://tempo2.sourceforge.net)
 * [libstempo](https://github.com/vallis/mc3pta/tree/master/stempo)
 * PyQt (see below)
 * Qt
 * IPython >= 2.0
 * pygments
 * pyzmq
 * jdcal
 * pyephem

PyQt on OSX
===========
Installing PyQt on OSX can best be done with macports or homebrew. If done with
homebrew however, be aware that you need to add the libraries to your path by
adding the following line to your .profile:

export PYTHONPATH=/usr/local/lib/python2.7/site-packages:$PYTHONPATH


Background info
===============
See more about binary models at on the homepage of [tempo](http://tempo.sourceforge.net/ref_man_sections/binary.txt)

Contact
=======
 * [_Rutger van Haasteren_](mailto:vhaasteren@gmail.com)

