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
import numpy as np

from PyQt4.QtGui import QApplication
from guidata.dataset.datatypes import (DataSet)
from guidata.dataset.dataitems import (IntItem, DirectoryItem, FloatArrayItem,
                                       StringItem)
import guidata.dataset.datatypes as dt
import guidata.dataset.dataitems as di

### Logging
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
debug, info, error = logging.debug, logging.info, logging.error

### Constants

### Classes

def parameter_object(cfg=None):

    def got_config():
        class ProcessingCfg(DataSet):
            # Set the starting value with the config file if given
            _dir = DirectoryItem("Shapefile Directory", cfg['user']['directory'])
            out_suffix = StringItem("Smoothed files suffix", cfg['user']['out_suffix'])
            out_dir = DirectoryItem("Output Directory", cfg['user']['out_dir'])
            buf_val_array = FloatArrayItem("Buffer Values",
                                           default=cfg['user']['buffer_value_array_km'],
                                           help="Units [km]",
                                           transpose=True)
            n_procs = IntItem("Number of Processors", min=0, max=10, default=cfg['system']['n_procs'])

        return ProcessingCfg()

    def no_config():
        class ProcessingNoCfg(DataSet):
            start_dir = os.getcwd()
            _dir = DirectoryItem("Shapefile Directory", start_dir)
            out_suffix = StringItem("Smoothed files suffix", "_b{buf_val}km_sm.shp")
            out_dir = DirectoryItem("Output Directory", start_dir)
            buf_val_array = FloatArrayItem("Buffer Values",
                                           default=np.array([10]),
  #                                             default=np.array([1, 5, 10, 50, 100]),
                                           help="Units [km]",
                                           transpose=True)
            n_procs = IntItem("Number of Processors", min=0, max=10, default=4)

        return ProcessingNoCfg()

    if cfg:
        params = got_config()
    else:
        params = no_config()
    return params

    class Processing(DataSet):
        """Arc Smooth"""
        # Set the starting value with the config file if given
        if cfg:
            _dir = DirectoryItem("Shapefile Directory", cfg['user']['directory'])
            out_suffix = StringItem("Smoothed files suffix", cfg['user']['out_suffix'])
            out_dir = DirectoryItem("Output Directory", cfg['user']['out_dir'])
            buf_val_array = FloatArrayItem("Buffer Values",
                                           default=cfg['user']['buffer_value_array_km'],
                                           help="Units [km]",
                                           transpose=True)
            n_procs = IntItem("Number of Processors", min=0, max=10, default=cfg['system']['n_procs'])
        else:
            start_dir = os.getcwd()
            _dir = DirectoryItem("Shapefile Directory", start_dir)
            out_suffix = StringItem("Smoothed files suffix", "_b{buf_val}km_sm.shp")
            out_dir = DirectoryItem("Output Directory", start_dir)
            buf_val_array = FloatArrayItem("Buffer Values",
                                           default=np.array([10]),
      #                                             default=np.array([1, 5, 10, 50, 100]),
                                           help="Units [km]",
                                           transpose=True)
            n_procs = IntItem("Number of Processors", min=0, max=10, default=4)
    return Processing()

class Processing2(dt.DataSet):
    """Example"""
    a = di.FloatItem("Parameter #1", default=2.3)
    b = di.IntItem("Parameter #2", min=0, max=10, default=5)
    type = di.ChoiceItem("Processing algorithm",
                         ("type 1", "type 2", "type 3"))





### Functions

### Tests

if __name__ == "__main__":



    print("Done __main__")

