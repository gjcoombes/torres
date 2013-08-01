# -*- coding: utf-8 -*-
"""
@author: G Coombes
Created on Sat Feb 02 13:21:36 2013
Script: arc_process_pkg2.py
Version: 0.5.5
Description:
A routine to smooth shapefile to the smae standard as the manual ArcMap method

Feature requests
1. Parse type from filename better
"""
from __future__ import print_function, division
import logging
import itertools
import json
import multiprocessing
import os
import pdb
import re
import shutil
import sys
import time
import ConfigParser
from fiona import collection
from itertools import repeat, izip, dropwhile, groupby
from collections import OrderedDict
from shapely.geometry import Polygon, MultiPolygon, mapping
from shapely.ops import unary_union
import numpy as np
from PyQt4.QtGui import QApplication

#guidata.qapplication() # not required if a QApplication has already been created

from guidata.dataset.datatypes import (DataSet)
from guidata.dataset.dataitems import (IntItem, DirectoryItem, FloatArrayItem,
                                       StringItem)

#logging.basicConfig(filename='arc_smooth.log', filemode='w', level=logging.DEBUG,
#                    format="%(asctime)s:%(levelname)s: %(message)s")
logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s:%(levelname)s: %(message)s")

def db(msg):
    logging.debug(msg)

def info(msg):
    logging.info(msg)

def must_process(shp_fp):
    """Predicate - should this file be processed?
    Also check shp_fp is a full path and exists
    Args:
        shp_fp: <str> complete path to a shapefile
    Returns:
       <bool> True if this file needs smoothing
    """
    logging.debug("Shapefile is {}".format(shp_fp))
    is_full_path = os.path.isabs(shp_fp)
    is_file = os.path.isfile(shp_fp)
    is_shape = os.path.splitext(shp_fp)[1] == '.shp'
    not_smoothed = "_sm" not in os.path.split(shp_fp)[1]
    return all([is_full_path, is_file, is_shape, not_smoothed])

def shp_files(_dir):
    """Return an iterator of shapefiles to process.

    # Equivalent to generator expression below
    all_files = []
    for p, _, fs in os.walk(_dir):
        for f in fs:
            all_files.append(os.path.join(p, f))
    """
    logging.debug("Inside func:shp_files")
    logging.debug("_dir is {}".format(_dir))
    all_files = ( os.path.join(p, f) for p, _, fs in os.walk(_dir) for f in fs )
    return ( f for f in all_files if must_process(f) )

def get_shp_file_fx(shp_fp, cfg=None):
    """Return a dict data structure
    *Outside world*

    Args:
        shp_fp: <str> A full path to the shapefile (assumes .shx, .dbf, .prj too)

    Returns:
        <dict> Mapping of all polygons in geo-json format
    """
    logging.debug("Inside func:get_shp_file_fx")
    return collection(shp_fp, 'r')

def parse_shp_type(shp_fp, cfg=None):
    """
    Args:
        shp_fp: <str> Absolute path

    Returns:
        _type: <str> Code to choose the type of processing required for each file

    For example:
        'surf_zone': 'Surface_Exposure_Zones'
        'entr_prob': 'Entrained_Exposure_Probability'
    _type has two elements
    (1) the oil_type - surface, shoreline, entrained, aromatic
    (2) the map type - probability, time, zone
    """
    logging.debug("Inside func:parse_shp_type")
    # Here are the regular expressions
    oil_patt_regex = 'surf|shor|entr|arom'
    map_patt_regex = 'prob|time|zone|dose|max'
    # Choose the filename portion of the filepath
    filename = os.path.split(shp_fp)[1]
    # Parse the oil type
    oil_patt = re.compile(oil_patt_regex, re.IGNORECASE)
    oil = oil_patt.search(filename)
    # Parse the  map_type
    map_patt = re.compile(map_patt_regex, re.IGNORECASE)
    map_ = map_patt.search(filename)
    # Check there is a match for each regex or raise an exception
    if oil:   # Is there a match?
        oil_str = oil.group().lower()
    else:
        err_str = "There are appears to be no match for the regex: {} in the string {}".format(
                oil_patt_regex, filename)
        raise ValueError(err_str)
    if map_:   # Is there a match?
        map_str = map_.group().lower()
    else:
        err_str = "There are appears to be no match for the regex: {} in the string {}".format(
                map_patt_regex, filename)
        raise ValueError(err_str)
    _type = "_".join([oil_str, map_str])
    logging.debug("_type is {}".format(_type))
    return _type


def calculate_key(record, _type, prop_key, cuts):
    log = lambda s: db("calculate_key: {}".format(s))
    PROPORTION_TO_PERCENTAGE = 100    ### STATE
    KG_M2_TO_G_M2 = 1000              ### STATE
    prob = record['properties'][prop_key]
    log("prob is {}".format(prob))
    if _type in ['surf_prob', 'entr_prob', 'arom_prob', 'shor_prob']:
        prob *= PROPORTION_TO_PERCENTAGE
    elif _type == 'surf_zone':
        prob *= KG_M2_TO_G_M2 #
    # Discarding value below first threshold for zones
    if _type in ['surf_zone', 'entr_zone', 'arom_zone']:
        if prob < cuts[1]:
            prob = 0
    if _type == 'surf_time':
        first_cut_func = first_cut_forwards
    else:
        first_cut_func = first_cut_backwards

    first_cut = first_cut_func(prob, cuts)
    db("First cut is {}".format(first_cut))
    return first_cut

def first_cut_backwards(prob, cuts):
    return next(dropwhile(lambda cut: cut > prob, cuts[::-1]))

def first_cut_forwards(prob, cuts):
    return next(dropwhile(lambda cut: cut <= prob, cuts))

def add_key_field(shp_coll, _type, cfg=None):
    """Add key for each polygon eg 5 for surf_prob.
    *Mutation of shp_coll*

    The key will the the bin that each polygon will be sorted in.
    For example, a polygon with a conc
    Args:
        shp_coll: A fiona.collection with add _type attribute

    Returns:
        (rec_list, _type)
            rec_list : <list> List of records with field 'key' added to each
            _type    : <str>
    """
    log = lambda s: db("add_key_field: {}".format(s))
    logging.debug("Inside func:add_key_field")


    prop_dict = cfg['config']['prop_dict']
    assert isinstance(prop_dict, dict)
    cuts_dict = cfg['config']['cuts_dict']
    assert isinstance(cuts_dict, dict)

    prop_key = prop_dict[_type]
    cuts = cuts_dict[_type]

    log("_type is {}".format(_type))
    log("prop_key is {}".format(prop_key))
    log("cuts are {}".format(cuts))

    coll_ls = []
    for record in shp_coll:
        key = calculate_key(record, _type, prop_key, cuts)
        # Discards polygons woth value of zero (arcmap null value)
        log("key is {}".format(key))
        record['properties']['key'] = key
        record['properties']['_type'] = _type
        coll_ls.append(record)
    log("Leaving func:add_key  **********************************")
    return coll_ls

def sort_and_partition(rec_list, cfg=None):
    """Sort by key and split into layers of the same key

    Args:
        rec_list <list> Seq of polygon records

    Returns:
        <dict: key: key, value: <list: of records>>
        A list of layers, each partitioned by key
    """
    logging.debug("Inside func:sort_and_partition")
    def key_func(elem):
        return elem['properties']['key']

    rec_list.sort(key=key_func)
    return {k: list(v) for k, v in groupby(rec_list, key=key_func)}

def smooth_layers_dict(layers_dict, buf_val=5, cfg=None):
    """Smooth the layers by key value -

    Called by func:run_job
    This function wraps the smooth_layer func.  It is here that some
    multiprocessing would benefit times (dissolve or cascaded union is the slowest
    part). Gather the smoothed layers here too.

    Args:
        layers_dict: A dictionary of lists, keyed by the binned value

    Returns:
        smoothed_dict: A dictionary of lists like layers_dict but with each
        layer buffered and dissolved
    """
    log = lambda s: db("smooth_layes_dict: {}".format(s))
    smoothed_dict = {}
    for key in sorted(layers_dict.keys()):
        log("Key is {}".format(key))
        smoothed_dict[key] = smooth_layer(layers_dict[key], buf_val, cfg=cfg)
    return smoothed_dict

def smooth_layer(record_ls, buf_val,
                 scale_km_to_degrees=0.009, delta_km=-0.5,     ### STATE
                 cfg=None):
    """Buffer out, dissolve and buffer back
    *Mutation*

    Args:
        record_ls: <list> A list of fiona records (from fiona collection)
        buf_val: <float> The value to buffer out each polygon, units of km
        scale_km_to_degrees <float>: Conversion based on *some* latitude
        delta_km <float>: Diference between buffer in and buffer out values,
            A -ve delta indicates the buffer_in is maller than the buffer_out

    Returns:
        <coll of shapely Polygons>
    """
    assert isinstance(buf_val, (float, int))
    b_out = buf_val * scale_km_to_degrees    ### STATE
    b_in = -(b_out + delta_km * scale_km_to_degrees)
    vert_ls = [ r['geometry']['coordinates'][0] for r in record_ls ]
    # Transform to shapely Polygons and guards empty polygons
    polygons = ( Polygon(v) for v in vert_ls if len(v) > 3 )
    dilated = ( p.buffer(b_out) for p in polygons if p.is_valid )
    dissolved = unary_union(list(dilated))
    eroded = dissolved.buffer(b_in)
    if isinstance(eroded, Polygon):
        eroded = MultiPolygon([eroded])
    logging.debug("Leaving func:smooth_layer")
    return eroded

def clip_layers(smoothed_dict, _type, cfg=None):
    """Clip layers from largest key to smallest
    *Mutation*

    The difference function a.difference(b) clips the area of b out of a.
    So we want to clip the smallest layer out of the second smallest layer
    second_smallest.difference(smallest)

    For probs ie 'surf_prob', 'entr_prob' we need to clip 100 from 90
    then 90 from 80
    For zones we clip 25 from 10, 10 from 1
    For times we clip 6 from 12, 12 from 24
    So times are clipped in ascending order, all other in descending key order

    Args:
        smoothed_dict of lists - I think each list is a MultiPolygon
    Returns:
        smoothed_dict with inner rings clipped out of outer rings
    """
    log = lambda s: db("clip_layers: {}".format(s))
    log("_type is {}".format(_type))
    if _type in ['surf_prob', 'entr_prob', 'arom_prob',
                'surf_zone', 'entr_zone', 'arom_zone',
                'shor_prob', 'shor_max']:
        reversed = True
    elif _type in ['surf_time']:
        reversed = False
    else:
        print("Warning, _type not found")

    keys = list(sorted(smoothed_dict.keys(), reverse=reversed))
    clipped_dict = dict.fromkeys(smoothed_dict, [])
    inner_keys = []
    for i in range(len(keys) - 1):
        inner_keys.append(keys[i])
        next_key = keys[i+1]
        assert isinstance(smoothed_dict[next_key], MultiPolygon)
        clipped_layer = smoothed_dict[next_key]
        for key in inner_keys:
            clipped_layer = clipped_layer.difference(smoothed_dict[key])
        clipped_dict[next_key] = clipped_layer
    clipped_dict[keys[0]] = smoothed_dict[keys[0]]
    logging.debug("Leaving func:clip_layers")
    return clipped_dict

def make_record(polygon, key, key_str, scale, _type):
    # Closure over key, key_str, scale and _type
    base_record = {'geometry': None,
               'properties': {}}
    record = base_record.copy()
    record['geometry'] = mapping(polygon)
    new_val = float(key*scale)
    record['properties'].update({key_str: new_val, '_type': _type})
    return record

def write_layers(clipped_dict, new_fp, _type, schema=None, cfg=None):
    """Write to new shapefile"""
#    logging.debug("Inside func:write_layers")
    log = lambda s: info("write_layers: {}".format(s))
    logging.info("Inside func:write_layers")
    key_str_dict = cfg['config']['key_str_dict']
    schema_dict = cfg['config']['schema_dict']
    out_key_str_dict = cfg['config']['out_key_str_dict']

    if schema is None:
        db("Setting schema")
        schema = schema_dict[_type]
        logging.debug("Schema is {}".format(schema))
    key_str = out_key_str_dict[_type]
    db("key_str is {}".format(key_str))

    # Rescale the key to represent original units
    PERCENT_TO_PROPORTION = 0.01    ### STATE
    G_M2_TO_KG_M2 = 0.001           ### STATE
    scale_dict = {'surf_prob': PERCENT_TO_PROPORTION,
                  'shor_prob': PERCENT_TO_PROPORTION,
                  'entr_prob': PERCENT_TO_PROPORTION,
                  'arom_prob': PERCENT_TO_PROPORTION,
                  'surf_zone': G_M2_TO_KG_M2}
    scale = scale_dict.get(_type, 1)

    driver = "ESRI Shapefile"
    logging.info("Writing to {}".format(new_fp))
    with collection(new_fp, 'w', driver, schema) as sink:
        for key, layer in sorted(clipped_dict.iteritems()):
            log("key is {} and scale is {}".format(key, scale))
            if isinstance(layer, Polygon):
                record = make_record(layer, key, key_str, scale, _type)
                db(record['properties'])
                sink.write(record)
            elif isinstance(layer, MultiPolygon):
                for polygon in layer.geoms:
                    record = make_record(polygon, key, key_str, scale, _type)
                    sink.write(record)
        sink.close()

def choose_n_procs(cfg):
    return cfg['system']['n_procs']

def default_config():
    cfg = OrderedDict([("user", OrderedDict([
               ("_comment",  "These are the variables you must enter"),
               ("directory",  r"C:\WinPython-64bit\apasa_src\arc_smooth\test_process"),
               ("buffer_value_array_km", [10]),
               ("smoothed_file_suffix", "_b{buf_val}km_sm.shp"),
#               ("buffer_value_array_km" , [10])
               ])),
          ("config", OrderedDict([
              ("_comment", "These are the values you can tweak, but should be ok left as is"),
              ("cuts_dict", {
                  "_comment": """Each cut value represents the maximum contained in the produced layer. For zone data, the first non-zero cut is discarded""",
                  'surf_prob' : [0, 5, 10, 25, 50, 75, 100],
                  'shor_prob' : [0, 5, 10, 25, 50, 75, 100],
                  'surf_time' : [0, 1, 6, 12, 24, 48, 120, 240, 480, 960, 1440, 1920, 2400, 2520, 10**9],
                  'surf_zone' : [0, 1, 10, 25, 10**9],
                  'entr_prob' : [0, 5, 10, 25, 50, 75, 100],
                  'entr_zone' : [0, 11520, 33600, 338400, 3859200, 10**9],
                  'arom_prob' : [0, 5, 10, 25, 50, 75, 100],
                  'arom_zone' : [0, 576, 4800, 38400, 10**9]
                  }),
              ("prop_dict", OrderedDict([
                     ('_comment', "These map the internal _type to incoming layer property name"),
                     ('surf_prob', 'concprob'),
                     ('shor_prob', 'shoreprob'),
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
              ("n_procs", 1),
            ])
          )])
    return cfg

def copy_prj_file(fp, new_fp):
    """Copy the prj file and save with new filename"""
    old_prj = os.path.splitext(fp)[0] + ".prj"
    new_prj = os.path.splitext(new_fp)[0] + ".prj"
    if os.path.isfile(old_prj):
        shutil.copy(old_prj, new_prj)
        logging.debug("Copying from {} to {}".format(old_prj, new_prj))
    else:
        logging.info("No prj file found -{}".format(old_prj))
    return None

def run_job(fp, buf_val, cfg):
    """

    Called by func: send_multiprocessing
    Args:
        fp: Absolute path to the shape file for processing
        buf_val: Distance (km) to buffer out for smoothing

    Returns: None
    """
    logging.debug("Inside func:run_job")

    suffix = cfg['user']['out_suffix']
    out_dir = cfg['user']['out_dir']
    print("out_dir is {}".format(out_dir))
    old_filename = os.path.split(fp)[1]
    file_stem = os.path.splitext(old_filename)[0]
    new_filename = file_stem + suffix.format(buf_val=buf_val)
    new_fp = os.path.join(out_dir, new_filename)
    print("New filename is {}".format(new_fp))
    with get_shp_file_fx(fp, cfg=cfg) as source:
        _type = parse_shp_type(source.name, cfg)
        src_gen = add_key_field(source, _type, cfg)

    layers_dc = sort_and_partition(src_gen, cfg)
    smoothed_dict = smooth_layers_dict(layers_dc, buf_val=buf_val, cfg=cfg)
    clipped_dict = clip_layers(smoothed_dict, _type, cfg=cfg)
    write_layers(clipped_dict, new_fp, _type, cfg=cfg)
    copy_prj_file(fp, new_fp)
    return new_fp

def send_multiprocessing(cfg):
    filenames = list(shp_files(cfg['user']['directory']))
    logging.info("Filenames are {}".format(map(str,filenames)))
    logging.info("Number of files for processing is {}".format(len(filenames)))
    buf_vals = cfg['user']['buffer_value_array_km']
    job_arg_ls = list(itertools.product(filenames, buf_vals, [cfg]))
    n_procs = cfg['system']['n_procs']
    pool = multiprocessing.Pool(n_procs)
    start = time.time()
    for job_args in job_arg_ls:
#        print("Job args are {}".format(job_args))
        pool.apply_async(run_job, job_args)
        print("sending job {}".format(job_args[:2]))
    pool.close()
    pool.join()
    n_jobs = len(job_arg_ls)
    elapsed_time_min = (time.time() - start) / 60
    return (n_jobs, elapsed_time_min, n_procs)

def send_single_job(cfg):
    filenames = list(shp_files(cfg['user']['directory']))
    logging.info("Filenames are {}".format(map(str,filenames)))
    logging.info("Number of files for processing is {}".format(len(filenames)))
    buf_vals = cfg['user']['buffer_value_array_km']
    job_arg_ls = list(itertools.product(filenames, buf_vals, [cfg]))
    start = time.time()
    f, b, c = job_arg_ls[0]
    print(b)
    print(type(b))
    assert isinstance(b, (float, int))
    run_job(*job_arg_ls[0])
    n_jobs = len(buf_vals)
    elapsed_time_min = (time.time() - start) / 60
    n_procs = 1
    return (n_jobs, elapsed_time_min, n_procs)

def mock_job(fp, buf_val):
    time.sleep(1)
    new_fn = os.path.splitext(fp)[0] + "b{}_sm.shp".format(buf_val)
    print("New filename is {}\n".format(os.path.split(new_fn)[1]))

def parse_cmd_args(cmd_args):
    """Determine if calling a gui, a file or a string"""
    logging.debug("Command args are {}".format(cmd_args))
    if cmd_args:
        first = cmd_args[0]
        try:
            test_val = json.loads(first)
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

def get_main_func(call_mode):
    def json_str_func(args):
        json_str = args[0]
        cfg = json.loads(json_str)
        success = send_multiprocessing(cfg)
#        success = send_single_job(cfg)
        return (success, cfg)

    def json_file_func(args):
        json_fp = args[0]
        with open(json_fp, "r") as fh:
            cfg = json.loads(fh.read())
#        success = send_multiprocessing(cfg)
        success = send_single_job(cfg)
        return (success, cfg)

    def ini_file_func(args):
        ini_fp = args[0]
        config = ConfigParser.SafeConfigParser()
        with open(ini_fp, "r") as fh:
            config.readfp(fh)
        cfg = config._sections
        # TODO Remove dirty eval GC 2013-02-08
        cfg['user']['buffer_value_array_km'] = eval(cfg['user']['buffer_value_array_km'])
        cfg['config']['cuts_dict'] = eval(cfg['config']['cuts_dict'])
        cfg['config']['prop_dict'] = eval(cfg['config']['prop_dict'])
        cfg['config']['key_str_dict'] = eval(cfg['config']['key_str_dict'])
        cfg['config']['schema_dict'] = eval(cfg['config']['schema_dict'])
        cfg['system']['n_procs'] = eval(cfg['system']['n_procs'])
        assert os.path.isdir(cfg['user']['directory']) ,\
                "Can't find the directory {}".format(cfg['user']['directory'])
        assert isinstance(cfg['config']['cuts_dict'], dict)
        assert isinstance(cfg['config']['prop_dict'], dict)
        assert isinstance(cfg['config']['key_str_dict'], dict)
        assert isinstance(cfg['config']['schema_dict'], dict)
#        success = send_multiprocessing(cfg)
        success = send_single_job(cfg)
        return (success. cfg)

    def gui_func(args, cfg=None):
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

        app = QApplication([])
        param = Processing()
        if param.edit(size=(600, 300)):
            cfg = default_config()
            param = remove_sentinel_buf_vals(param)
            cfg['user']['directory'] = param._dir
            cfg['user']['buffer_value_array_km'] = param.buf_val_array
            cfg['user']['out_suffix'] = param.out_suffix
            cfg['system']['n_procs'] = param.n_procs
            cfg['user']['out_dir'] = param.out_dir
#            success = send_multiprocessing(cfg)
            success = send_single_job(cfg)
            return (success, cfg)
        else:
            return None

    def remove_sentinel_buf_vals(param):
        tmp_ls = []
        SENTINEL = -1
        for elem in param.buf_val_array.tolist():
            if not elem == SENTINEL:
                tmp_ls.append(elem)
        logging.debug("Are the -1 removed?: {}".format(tmp_ls))
        param.buf_val_array = tmp_ls
        return param

    func_dict = {'json_str': json_str_func,
                 'json_file': json_file_func,
                 'ini_file': ini_file_func,
                 'gui': gui_func }
    # Return the function that main will call
    return func_dict[call_mode]

def main(cmd_args):
    # 1. Determine if calling a gui, a file or a string
    call_mode = parse_cmd_args(cmd_args)
    # 2. Call appropriate func
    main_func = get_main_func(call_mode)
    # 3 Exit with success or cancel
    success = True
    print("main_func_name is {}".format(main_func.__name__))
    if main_func.__name__ == "gui_func":
        cfg = None
        while success:
            success, cfg = main_func(cmd_args, cfg)
            if success:
                n_jobs, elapsed_time_min, n_procs = success
                print("{} shapefiles created in {:.1f} minutes with {} processors".format(
                    n_jobs, elapsed_time_min, n_procs))
    else:
        success, cfg = main_func(cmd_args)
        if success:
            n_jobs, elapsed_time_min, n_procs = success
            print("{} shapefiles created in {:.1f} minutes with {} processors".format(
                n_jobs, elapsed_time_min, n_procs))

    return None

if __name__ == "__main__":

#    main(sys.argv[1:])
#    main(['..\apasa_src\arc_smooth\test_config.ini'])
    main([r'test_config.json'])

    print('Done')
