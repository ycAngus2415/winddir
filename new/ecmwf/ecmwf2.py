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
    'date'      : "2016-01-01/to/2016-02-29",
    'type'      : "an",
    'class'     : "ei",
    'area'      : "60/-135/10/-100",
    'format'    : "netcdf",
    'target'    : "interim_west_01.nc"
 })
