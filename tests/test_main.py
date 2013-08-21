#! /usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Module: test_main.py
Created on Wed Aug 21 13:27:13 2013
@author: gcoombes
Description:

"""
### Imports
from __future__ import print_function
from __future__ import division

import os.path as osp

import pytest
from context import torres
from torres.main import *
### Logging
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
debug, info, error = logging.debug, logging.info, logging.error

### Constants

### Classes

### Functions

### Tests
def test_parse_cmd_args_gui():
    call_mode = parse_cmd_args(["monkeys"])
    assert call_mode == "gui"

def test_parse_cmd_args_json_str():
    call_mode = parse_cmd_args(['{"animal": "monkeys"}'])
    assert call_mode == "json_str"

def test_parse_cmd_args_json_file():
    call_mode = parse_cmd_args(['test.json'])
    assert call_mode == "json_file"

def test_parse_cmd_args_ini_file():
    call_mode = parse_cmd_args(['test.ini'])
    assert call_mode == "ini_file"

def test_parse_cmd_args_dir():
    dir_ = osp.abspath(".")
    call_mode = parse_cmd_args([dir_])
    assert call_mode == "dir"




if __name__ == "__main__":
    pytest.main(['test_main.py', '-xvv'])
#    pytest.main(['test_main.py', '--cov-report', 'html', '--cov', '.', '-xvv'])

    print("Done __main__")

