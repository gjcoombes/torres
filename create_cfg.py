# -*- coding: utf-8 -*-
"""
Created on Wed Feb 06 14:01:36 2013
script: create_cfg.py
@author: gcoombes
"""
import json
from pprint import pprint
import ConfigParser
from collections import OrderedDict
from torres.config.defaults import default_config


def make_json_file(json_fp=None):
    if not json_fp:
        json_fp = 'test_config.json'
    cfg = default_config()

    with open(json_fp, "w") as fh:
        fh.write(json.dumps(cfg, indent=2))
    return json_fp



def make_ini_file(ini_fp=None):
#print("**Config parsing ini files**")
    if not ini_fp:
        ini_fp = "test_config.ini"
    config = ConfigParser.RawConfigParser()

    result = default_config()
    with open(ini_fp, 'w') as fh:
        for section_key, section in result.iteritems():
            config.add_section(section_key)
            for key, value in sorted(section.iteritems()):
                config.set(section_key, key, str(value))
        config.write(fh)
    return(ini_fp)

make_json_file()
make_ini_file()
#config = ConfigParser.RawConfigParser()
#ini_fp = make_ini_file()
#with open(ini_fp, "r") as fh:
#    config.readfp(fh)
#
#d = config._sections
#print(type(d['user']))
#od = d['user']
#print(od['buffer_value_array_km'])
#print(type(od))
##with open
#print("Done")


















print("Done")