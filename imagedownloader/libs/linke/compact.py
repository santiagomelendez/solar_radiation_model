import glob
from osgeo import gdal
from libs.file import netcdf as nc

files = glob.glob("libs/linke/tifs/*.tif")
shape = gdal.Open(files[0]).ReadAsArray().shape

root, n = nc.open("linketurbidity.nc")
nc.getdim(root, "northing", shape[0])
nc.getdim(root, "easting", shape[1])
nc.getdim(root, "monthing")
linke = nc.getvar(root, "linketurbidity", "f4", ("monthing", "northing", "easting",), 4)

for i in range(len(files)):
	print i, '->', files[i]
	linke[i,:] = gdal.Open(files[i]).ReadAsArray()

nc.close(root)