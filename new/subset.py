import snappy
from snappy import ProductIO
import os
import calibrate
import gc

#filepath = '/Users/yangchao/GitHub/wind/download/'
filepath = '/Volumes/hwang/yc1/'

#filepath = '/Users/yangchao/GitHub/wind/snap/data/'
Height = 400
Width = 400

subsetop = snappy.jpy.get_type('org.esa.snap.core.gpf.common.SubsetOp')
wktreader = snappy.jpy.get_type('com.vividsolutions.jts.io.WKTReader')
rectangel = snappy.jpy.get_type('java.awt.Rectangle')
pixelpos = snappy.jpy.get_type('org.esa.snap.core.datamodel.PixelPos')
geopos = snappy.jpy.get_type('org.esa.snap.core.datamodel.GeoPos')


# cut in x,y with height and width
def subset(file, x, y):
    sbop = subsetop()
    sbop.setSourceProduct(file)
    # wkt = wktreder()
    # geometry = wkt.read(wkt)
    # sbop.setGeoRegion(geometry)
    sbop.setRegion(rectangel(x, y, Height, Width))
    sbop.setCopyMetadata(True)
    try:
        subfile = sbop.getTargetProduct()
    except Exception:
        subfile = False
    finally:
        del sbop
        return subfile


# get the match up wind filed
def wind_filed_from_ndbc(station, time):
    time_string = time[0:4]+' '+time[4:6]+' '+time[6:8]+' '+time[9:11]+' '+str(int(time[11:13])//10*10)
    with os.popen('cd ./ndbc \n grep "'+time_string+'" '+station+'*') as file:
        datas = file.readlines()
        if len(datas) == 0:
            return 0, 0
        else:
            for data in datas:
                wdir = data.split(':')[1][17:20]
                wspeed = data.split(':')[1][21:25]
            return wdir, wspeed


def get_all_station():
    with open('/Volumes/Yangchao/data/all_ndbc_information1.txt') as files:
        stations = {}
        lat = []
        lon = []
        height = []
        name = []
        for station in files:
            ss = station.split(' ')
            name.append(ss[0])
            if ss[2] == 'N':
                lat.append(float(ss[1]))
            else:
                lat.append(-float(ss[1]))
            if ss[4] == 'E':
                lon.append(float(ss[3]))
            else:
                lon.append(-float(ss[3]))
            if len(ss) > 12:
                height.append(ss[-5])
            else :
                height.append('5')
        stations['name'] = name
        stations['lat'] = lat
        stations['lon'] = lon
        stations['height'] = height
        return stations


# match up the data
def match_up(filename, file, stations):
    print('match_up...')
#    file = ProductIO.readProduct(filepath+filename)
    geo = file.getSceneGeoCoding()
    size = file.getSceneRasterSize()
    if geo.canGetGeoPos():
        for station in stations:
            p1 = geo.getPixelPos(geopos(station['Lat'], station['Lon']), None)
            if p1.getX() == 'non':
                continue
            x = p1.getX() - Height / 2
            y = p1.getY() - Width / 2
            if file.containsPixel(p1):
                flist = filename.split('_')
                wdir, wspeed = wind_filed_from_ndbc(station['name'], flist[-5])
                if wdir == 0 and wspeed == 0:
                    continue
                else:
                    print(flist[-1].split('.')[0])
                    print(station)
                    print(x, y)
                    print(size.getHeight(), size.getWidth())
                    if size.getHeight() > 400 and size.getWidth > 400:
                        subfile = subset(file, int(x), int(y))
                        if subfile is False:
                            print('cut exception')
                            continue
                        print('write...')
                        print(filename, station['name'])
                        try:
                            ProductIO.writeProduct(subfile, '/users/yangchao/GitHub/wind/snap/match_data1/'+station['name']+'_subset_'+wdir+'_'+wspeed+'_'+station['height']+'_'+filename.split('.')[0], 'BEAM-DIMAP')
                        except Exception:
                            print('exception')
                            subfile.dispose()
                            continue
                        subfile.dispose()
                        del subfile
                    else:
                        print('copy...')
                        try:
                            ProductIO.writeProduct(file, '/users/yangchao/GitHub/wind/snap/match_data1/'+station['name']+'_subset_'+wdir+'_'+wspeed+'_'+station['height']+'_'+filename.split('.')[0], 'BEAM-DIMAP')
                        except Exception:
                            print('exception')
                            continue
    else:
        print(filename + " can't get geo pos")

    del size
    del geo
    file.dispose()


if __name__ == '__main__':
    stations = get_all_station()
    print('my file')
    with os.popen('cd '+filepath+' \n ls *.zip') as files:
        i = 1
        for filename in files:
            print(format('my file %d'%i))
            i += 1
            sourceProduct = ProductIO.readProduct(filepath+filename[:-1])
            file = calibrate.thermal_app(filename[:-1], sourceProduct)
            match_up(filename[:-1], file, stations)
            sourceProduct.dispose()
            sourceProduct.dispose()
            file.dispose()
            del file
            del sourceProduct
            print('-----------down----------')
            gc.collect()
#    print('from xhl')
#    with os.popen('cd /Volumes/new/Sentinel-data-Jialiu \n ls -a') as f:
#        i = 1
#        for ff in f:
#            if i<50:
#                i += 1
#                continue
#            print('from xhl %d'%i)
#            if len(ff) == 6:
#                print(ff)
#                with os.popen('cd '+'/Volumes/new/Sentinel-data-Jialiu/'+ff[:-1]+' \n ls *.zip') as files:
#                    for filename in files:
#                        sourceProduct = ProductIO.readProduct('/Volumes/new/Sentinel-data-Jialiu/'+ff[:-1]+'/'+filename[:-1])
#                        file = calibrate.thermal_app(filename[:-1], sourceProduct)
#                        match_up(filename[:-1], file, stations)
#                        sourceProduct.dispose()
#                        file.dispose()
#                        del file
#                        del sourceProduct
#                        print('-------down-----')
#                        gc.collect()
