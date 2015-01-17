import numpy as np
from netcdf import netcdf as nc
from libs.geometry import project as p
from datetime import datetime
from libs.console import say
import os
localpath = os.path.dirname(__file__)
gdal_supported = False
try:
	from osgeo import gdal
	gdal_supported = True
except Exception:
	pass

source = "http://limie.com.ar/gersolar_collab/linketurbidity.nc"
filename = source.split("/")[-1]
destiny = localpath + "/" + filename

def initial_configuration():
	from libs.file.toolbox import download
	say("Downloading "+filename+" package... ")
	download(source,destiny)

if not os.path.exists(destiny):
	initial_configuration()

def get_month(month):
    if gdal_supported:
        ds = gdal.Open(localpath + "/tifs/"+str(month).zfill(2)+"_longlat_wgs84.tif")
        linke = ds.ReadAsArray()
    else:
        root, _ = nc.open(destiny)
        data = nc.getvar(root, "linketurbidity")
        linke = data[0,month -1,:]
	# The linke turbidity is obtained when the image pixel value is divied by 20.
	return linke/20.

def project_coordinates(lat, lon):
	shape = get_month(1).shape
	return p.pixels_from_coordinates(lat, lon, shape[0], shape[1])

def cut_month(x, y, month):
	linke = get_month(month)
	result = p.transform_data(linke,x,y)
	return np.float32(result)

def cut_projected(root):
	lat = nc.getvar(root, 'lat')
	lon = nc.getvar(root, 'lon')
	time = nc.getvar(root, 'time')
	months = list(set([ (datetime.fromtimestamp(int(t))).month for t in time ]))
	nc.getdim(root, 'monthing')
	months_cut = root.getvar('months', 'i2', ('monthing',))
	dims=list(root.variables['lat'].dimensions)
	dims.insert(0, 'monthing')
	linke = root.getvar('linketurbidity', 'f4', tuple(dims),4)
	linke_x, linke_y = project_coordinates(lat[:], lon[:])
	months_cut[:] = np.array(list(months))
	for i in range(len(months)):
		linke[i] = cut_month(linke_x, linke_y, months[i])
	return linke[:], linke_x, linke_y
