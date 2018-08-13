import snappy
from snappy import ProductIO
import datetime
import sys
import os
import gc
subsetop = snappy.jpy.get_type('org.esa.snap.core.gpf.common.SubsetOp')
wktreader = snappy.jpy.get_type('com.vividsolutions.jts.io.WKTReader')
pixelpos = snappy.jpy.get_type('org.esa.snap.core.datamodel.PixelPos')
geopos = snappy.jpy.get_type('org.esa.snap.core.datamodel.GeoPos')


#init time is 1900 1, 1 , 0, 0, 0
init_time = datetime.datetime(1900, 1, 1,0,0,0)
file_list = ['east02.nc', 'west02.nc']

w = "POLYGON((-67.84593200683594 38.769126892089844, -58.153602600097656 38.769126892089844, -58.153602600097656 24.92294692993164, -67.84593200683594 24.92294692993164, -67.84593200683594 38.769126892089844))"
'''
subset image from ecmwf cosponding with s1-a
'''
def subset(files,  w, bandlist):
    wkt = wktreader()
    geometries = wkt.read(w)
    subop = subsetop()
    subop.setParameterDefaultValues()
    subop.setSourceProduct(files)
    subop.setGeoRegion(geometries)
    subop.setCopyMetadata(True)
    subop.setBandNames(bandlist)
    dd = subop.getTargetProduct()
    print('--------success-----')
    return dd

'''
write source file to object file
'''
def write_product(source_file, object_file):
    ProductIO.writeProduct(source_file, object_file,'BEAM-DIMAP')

'''
open source file
'''
def open_product(filename):
    files = ProductIO.readProduct(filename)
    return files

'''
get lon and lat of the source file
'''
def get_lon_lat(file):
    files = open_product(file)
    size = files.getSceneRasterSize()
    print('\n-----size------')
    print(size.getHeight(), size.getWidth())
    geocoding = files.getSceneGeoCoding()
    gp1 = geocoding.getGeoPos(pixelpos(0,0),None)
    gp2 = geocoding.getGeoPos(pixelpos(0,size.getHeight()),None)
    gp3 = geocoding.getGeoPos(pixelpos(size.getWidth(),0),None)
    gp4 = geocoding.getGeoPos(pixelpos(size.getWidth(),size.getHeight()),None)
    lon = []
    lon.append(gp1.lon)
    lon.append(gp2.lon)
    lon.append(gp3.lon)
    lon.append(gp4.lon)
    lat = []
    lat.append(gp1.lat)
    lat.append(gp2.lat)
    lat.append(gp3.lat)
    lat.append(gp4.lat)
    files.dispose()
    geocoding.dispose()
    print(' ')
    print('----lon and lat----')
    print(lon)
    print(lat)
    return (lon, lat)

'''
get polygon for wkt
'''
def get_polygon(file):
    lon, lat = get_lon_lat(file)
    s = 'POLYGON((%f %f, %f %f, %f %f, %f %f, %f %f))'%(lon[0], lat[0], lon[1], lat[1], lon[3], lat[3], lon[2], lat[2], lon[0], lat[0])
    print('\n------polygon-----')
    print(s)
    return s

'''
get time of the source files from file name
'''
def get_time(file):
    file_part_list = file.split('_')
    time1 = file_part_list[4]
    time_part = time1.split('T')
    date = time_part[0]
    time = time_part[1]
    year = int(date[:4])
    month = int(date[4:6])
    day = int(date[6:])
    hour = int(time[:2])
    minute = int(time[2:4])
    data_list = [year, month, day, hour, minute]
    print('\n------time------')
    print(data_list)
    return data_list

'''
get bandlist
'''
def get_bandlist(file, v):
    time= get_time(file)
    date1 = datetime.datetime(2015, 10,1,0,0,0)
    date2 = datetime.datetime(time[0], time[1], time[2], time[3], time[4],0)
    dd = date2-date1
    band = int(dd.total_seconds()/60/60/6)+1
    bandlist=['u10_time%d'%band, 'v10_time%d'%band]
    print('\n------bandlist------')
    print(bandlist)
    return bandlist

'''
selet target product to subset
1, interm_east_01
2, interm_west_01
3, interm_east
4, interm_west
'''
def get_target(file):
    date = get_time(file)
    lon, lat = get_lon_lat(file)
    #if the one point is in the image, we get it
    #east
    if lon[0]>-84 and lon[0] < -48 and lat[0] <50 and lat[0] >10:
            return 0
    #west
    elif lon[0]>-135 and lon[0]<-110 and lat[0] < 60 and lat[0]>28:
            return 1

'''
main
file = '../ecmwf/'
'''
if __name__ == '__main__':
    file_save_path = '../ecmwf/'
    file_path = '/Volumes/Temperament/yc/'
    filename = 'S1A_IW_GRDH_1SDV_20160303T222554_20160303T222623_010211_00F13B_D028'
    v = get_target(file_path+filename+'.zip')
    obj = file_save_path+file_list[v]
    obj_files = open_product(obj)
    band_list = get_bandlist(file_path+filename+'.zip', v)
    wkt = get_polygon(file_path+filename+'.zip')
    target = subset(obj_files, wkt, band_list)
    write_product(target, file_save_path+'match_'+filename+'.dim')
    obj_files.dispose()
    target.dispose()
    gc.collect()
