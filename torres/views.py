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
    if cfg is None:
        _out_dir = _start_dir = os.getcwd()
        _out_suffix = "_b{buf_val}km_sm.shp"
        _buf_val_array = np.array([1, 5, 10, 50, 100]),
#        _buf_val_array = np.array([10])
        _n_procs = 4
    else:
        _start_dir = cfg['user']['directory']
        _out_dir = cfg['user']['out_dir']
        _out_suffix = cfg['user']['out_suffix']
        _buf_val_array = cfg['user']['buffer_value_array_km']
        _n_procs = cfg['system']['n_procs']

    class Processing(DataSet):
        _dir = DirectoryItem("Shapefile Directory", _start_dir)
        out_suffix = StringItem("Smoothed files suffix", _out_suffix)
        out_dir = DirectoryItem("Output Directory", _out_dir)

        n_procs = IntItem("Number of Processors", min=0, max=10, default=_n_procs)
        g0 = dt.BeginTabGroup("group")
        buf_val_array = FloatArrayItem("Buffer Values",
                                       default=_buf_val_array,
                                       help="Units [km]",
                                       transpose=True)
        mchoice1 = di.MultipleChoiceItem("Smoothing",
                                      ["first choice", "second choice",
                                       "third choice"]).vertical(2)
        mchoice2 = di.ImageChoiceItem("Thresholds",
                                   [("rect", "first choice", "gif.png" ),
                                    ("ell", "second choice", "txt.png" ),
                                    ("qcq", "third choice", "file.png" )]
                                   ).set_pos(col=1) \
                                    .set_prop("display", icon="file.png")
        mchoice3 = di.MultipleChoiceItem("MC type 3",
                                      [ str(i) for i in range(10)] ).horizontal(2)
        eg0 = dt.EndTabGroup("group")


    return Processing()


#    class Processing(DataSet):
#        """Arc Smooth"""
#        # Set the starting value with the config file if given
#        if cfg:
#            _dir = DirectoryItem("Shapefile Directory", cfg['user']['directory'])
#            out_suffix = StringItem("Smoothed files suffix", cfg['user']['out_suffix'])
#            out_dir = DirectoryItem("Output Directory", cfg['user']['out_dir'])
#            buf_val_array = FloatArrayItem("Buffer Values",
#                                           default=cfg['user']['buffer_value_array_km'],
#                                           help="Units [km]",
#                                           transpose=True)
#            n_procs = IntItem("Number of Processors", min=0, max=10, default=cfg['system']['n_procs'])
#        else:
#            start_dir = os.getcwd()
#            _dir = DirectoryItem("Shapefile Directory", start_dir)
#            out_suffix = StringItem("Smoothed files suffix", "_b{buf_val}km_sm.shp")
#            out_dir = DirectoryItem("Output Directory", start_dir)
#            buf_val_array = FloatArrayItem("Buffer Values",
#                                           default=np.array([10]),
#      #                                             default=np.array([1, 5, 10, 50, 100]),
#                                           help="Units [km]",
#                                           transpose=True)
#            n_procs = IntItem("Number of Processors", min=0, max=10, default=4)
#    return Processing()

class Processing2(dt.DataSet):
    """Example"""
    a = di.FloatItem("Parameter #1", default=2.3)
    b = di.IntItem("Parameter #2", min=0, max=10, default=5)
    type = di.ChoiceItem("Processing algorithm",
                         ("type 1", "type 2", "type 3"))





### Functions

### Tests

if __name__ == "__main__":
    from torres.main import main
    main(sys.argv)


    print("Done __main__")

