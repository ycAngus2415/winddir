#!/usr/bin/env python
from ecmwfapi import ECMWFDataServer

server = ECMWFDataServer()

server.retrieve({
    'stream'    : "oper",
    'levtype'   : "sfc",
    'param'     : "165.128/166.128",
    'dataset'   : "interim",
    'step'      : "3/6/9/12",
    'grid'      : "0.125/0.125",
    'time'      : "00/06/12/18",
    'date'      : "2016-01-01/to/2016-02-28",
    'type'      : "an",
    'class'     : "ei",
    'area'      : "60/-84/10/-40",
    'format'    : "netcdf",
    'target'    : "interim_east_01.nc"
 })
