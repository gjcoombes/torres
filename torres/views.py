#! /usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Module: views.py
Created on Wed Aug 21 13:15:41 2013
@author: gcoombes
Description:

"""
### Imports
from __future__ import print_function
from __future__ import division

import sys, os

from PyQt4.QtGui import QApplication
from guidata.dataset.datatypes import (DataSet)
from guidata.dataset.dataitems import (IntItem, DirectoryItem, FloatArrayItem,
                                       StringItem)

### Logging
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
debug, info, error = logging.debug, logging.info, logging.error

### Constants

### Classes
class Processing(DataSet):
    """Arc Smooth"""
    def __init__(self, cfg=None):
    # Set the starting value with the config file if given
        if cfg:
            self._dir = DirectoryItem("Shapefile Directory", cfg['user']['directory'])
            self.out_suffix = StringItem("Smoothed files suffix", cfg['user']['out_suffix'])
            self.out_dir = DirectoryItem("Output Directory", cfg['user']['out_dir'])
            self.buf_val_array = FloatArrayItem("Buffer Values",
                                           default=cfg['user']['buffer_value_array_km'],
                                           help="Units [km]",
                                           transpose=True)
            self.n_procs = IntItem("Number of Processors", min=0, max=10, default=cfg['system']['n_procs'])
        else:
            self.start_dir = os.getcwd()
            self._dir = DirectoryItem("Shapefile Directory", start_dir)
            self.out_suffix = StringItem("Smoothed files suffix", "_b{buf_val}km_sm.shp")
            self.out_dir = DirectoryItem("Output Directory", start_dir)
            self.buf_val_array = FloatArrayItem("Buffer Values",
                                           default=np.array([10]),
      #                                             default=np.array([1, 5, 10, 50, 100]),
                                           help="Units [km]",
                                           transpose=True)
            self.n_procs = IntItem("Number of Processors", min=0, max=10, default=4)



### Functions

### Tests

if __name__ == "__main__":



    print("Done __main__")

