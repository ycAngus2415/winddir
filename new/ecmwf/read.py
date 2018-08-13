import netCDF4 as nc
from netCDF4 import Dataset
'''changing the longitude into the range -180 to 180'''
filename = ['./east02.nc', 'west02.nc']
for file in filename:

    files =Dataset(file, 'a')
    lon = files.variables['longitude']
    lon[:] = lon[:] - 360
    files.close()

