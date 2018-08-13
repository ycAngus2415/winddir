import snappy
from snappy import ProductIO
from snappy import PixelPos
from snappy import GeoPos
import datetime
import sys
import os
import gc
subsetop = snappy.jpy.get_type('org.esa.snap.core.gpf.common.SubsetOp')
wktreader = snappy.jpy.get_type('com.vividsolutions.jts.io.WKTReader')

def get_subset(files1, files2):
    files1_geocoding = files1.getSceneRasterGeoCoding()
    files2_geocoding = files2.getSceneRasterGeoCoding()
    height = files1.getSceneRasterHeight()
    width =files1.getSceneRasterWidth()
    for h in range(1, height):
        for w in range(1, width):
            g = files1_geocoding.getGeoPos(PixelPos(w, h), None)
            p = files2_geocoding.getPixelPos(g, None)
            if files2.containsPixel(p):




def contains(files, p1, p2, p3, p4):
    if files.containsPixel(p1) and files.containsPixel(p2) and files.containsPixel(p3) and files.containsPixel(p4):
        return True
    else:
        return False


