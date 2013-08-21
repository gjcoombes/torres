#! /usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Module: main.py
Created on Wed Aug 21 13:20:00 2013
@author: gcoombes
Description: Entry point for commandline

"""
### Imports
from __future__ import print_function
from __future__ import division

import sys, os
import json

import pytest

from torres import controllers

### Logging
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
debug, info, error = logging.debug, logging.info, logging.error

### Constants
func_dict = {
    "json_str": None,
    "dir": None,
    "json_file": None,
    "ini_file": None,
    "gui": controllers.gui,
}
### Classes

### Functions


def parse_cmd_args(cmd_args):
    """Determine if calling a gui, a file or a string"""
    logging.debug("Command args are {}".format(cmd_args))
    if cmd_args:
        first = cmd_args[0]
        try:
            _ = json.loads(first)
            call_mode = 'json_str'
        except ValueError:
            if os.path.isdir(first):
                call_mode = 'dir'
            elif os.path.splitext(first)[1] == ".json":
                call_mode = 'json_file'
            elif os.path.splitext(first)[1] == ".ini":
                call_mode = 'ini_file'
            else:
                call_mode = 'gui'
    else:
        call_mode = 'gui'
    return call_mode

def main(args):
    args = sys.argv[1:]
    call_mode = parse_cmd_args(args)
    main_func = func_dict[call_mode]
    main_func(args)

### Tests

if __name__ == "__main__":
#    pytest.main()
    pytest.main([r'..\tests\test_main.py', '--cov-report', 'html', '--cov', '.', '-xvv'])
    main(sys.argv)
    print("Done __main__")

