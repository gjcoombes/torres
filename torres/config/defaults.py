#! /usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Module: defaults.py
Created on Wed Aug 21 14:22:05 2013
@author: gcoombes
Description:

"""
### Imports
from __future__ import print_function
from __future__ import division

from collections import OrderedDict

### Logging
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
debug, info, error = logging.debug, logging.info, logging.error

### Constants

### Classes

### Functions
def default_config():
    cfg = OrderedDict([("user", OrderedDict([
               ("_comment",  "These are the variables you must enter"),
               ("directory",  r"G:\torres\to_process"),
#               ("buffer_value_array_km", [1, 5, 10, 50, 100]),
               ("out_suffix", "_b{buf_val}km_sm.shp"),
               ("buffer_value_array_km" , [10]),
               ("out_dir", r"G:\torres\out"),
               ])),
          ("config", OrderedDict([
              ("_comment", "These are the values you can tweak, but should be ok left as is"),
              ("cuts_dict", {
                  "_comment": """Each cut value represents the maximum contained in the produced layer. For zone data, the first non-zero cut is discarded""",
                  'surf_prob' : [0, 5, 10, 25, 50, 75, 101],
                  'shor_prob' : [0, 5, 10, 25, 50, 75, 101],
                  'shor_max'  : [0, 1, 10, 25, 10**9],
                  'surf_time' : [0, 1, 6, 12, 24, 48, 120, 240, 360, 480, 600, 720, 960, 1440, 1920, 2352, 5000],
                  'surf_zone' : [0, 1, 10, 25, 10**9],
                  'entr_prob' : [0, 5, 10, 25, 50, 75, 101],
                  'entr_zone' : [0, 11520, 33600, 338400, 3859200, 10**9],
                  'arom_prob' : [0, 5, 10, 25, 50, 75, 101],
                  'arom_zone' : [0, 576, 4800, 38400, 10**9]
                  }),
              ("prop_dict", OrderedDict([
                     ('_comment', "These map the internal _type to incoming layer property name"),
                     ('surf_prob', 'concprob'),
                     ('shor_prob', 'shoreprob'),
                     ('shor_max', 'shore_max'),
                     ('surf_time', 'surfthresht'),
                     ('surf_zone', 'conc_max'),
                     ('entr_prob', 'wcdoseprob'),
                     ('entr_zone', 'wcmaxdose'),
                     ('arom_prob', 'aromdosepro'),
                     ('arom_zone', 'arommaxdose'),
                     ])),
              ("key_str_dict", OrderedDict([
                     ('_comment', "These map the internal _type to outgoing layer property name"),
                     ('surf_prob', 'concprob'),
                     ('shor_prob', 'shoreprob'),
                     ('shor_max', 'shore_max'),
                     ('surf_time', 'surfthresht'),
                     ('surf_zone', 'conc_max'),
                     ('entr_prob', 'wcdoseprob'),
                     ('entr_zone', 'wcmaxdose'),
                     ('arom_prob', 'aromdosepro'),
                     ('arom_zone', 'arommaxdose'),
                     ])),
              ("out_key_str_dict", OrderedDict([
                     ('_comment', "These map the internal _type to outgoing layer property name"),
                     ('surf_prob', 'concprob'),
                     ('shor_prob', 'shoreprob'),
                     ('shor_max', 'shore_max'),
                     ('surf_time', 'surfthresh'),
                     ('surf_zone', 'conc_max'),
                     ('entr_prob', 'wcdoseprob'),
                     ('entr_zone', 'wcmaxdose'),
                     ('arom_prob', 'aromdosepr'),
                     ('arom_zone', 'arommaxdos'),
                     ])),
              ("schema_dict", OrderedDict([
                     ('_comment', 'This describes the schema of the ESRI shapefile'),
                     ('surf_prob', {'geometry':'Polygon','properties': {'concprob': 'float','_type': 'str'}}),
                     ('shor_prob', {'geometry':'Polygon','properties': {'shoreprob': 'float','_type': 'str'}}),
                     ('shor_max',  {'geometry':'Polygon','properties': {'shore_max': 'float','_type': 'str'}}),
                     ('surf_time', {'geometry':'Polygon','properties': {'surfthresh': 'float','_type': 'str'}}),
                     ('surf_zone', {'geometry':'Polygon','properties': {'conc_max': 'float','_type': 'str'}}),
                     ('entr_prob', {'geometry':'Polygon','properties': {'wcdoseprob': 'float','_type': 'str'}}),
                     ('entr_zone', {'geometry':'Polygon','properties': {'wcmaxdose': 'float','_type': 'str'}}),
                     ('arom_prob', {'geometry':'Polygon','properties': {'aromdosepr': 'float','_type': 'str'}}),
                     ('arom_zone', {'geometry':'Polygon','properties': {'arommaxdos': 'float','_type': 'str'}}),
                     ])),
                ])),
          ("system", OrderedDict([
              ("_comment", "This for meta-setup details"),
              ("n_procs", 2),
            ])
          )])
    return cfg
### Tests

if __name__ == "__main__":



    print("Done __main__")

