#!/usr/bin/env python
from ecmwfapi import ECMWFDataServer

server = ECMWFDataServer()

server.retrieve({
    'stream'    : "oper",
    'levtype'   : "sfc",
    'param'     : "165.128/166.128",
    'dataset'   : "interim",
    'step'      : "0",
    'grid'      : "0.125/0.125",
    'time'      : "00/06/12/18",
    'date'      : "2015-11-01/to/2017-12-31",
    'type'      : "an",
    'class'     : "ei",
    'area'      : "50/-84/10/-48",
    'format'    : "netcdf",
    'target'    : "east02.nc"
 })
