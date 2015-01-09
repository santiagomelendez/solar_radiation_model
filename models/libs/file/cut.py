import sys
sys.path.append('.')
import numpy as np
import netcdf as nc
from libs.console import show, say

def statistical_search_position(station, lat, lon):
	diffs = np.abs(lat[:] - station[0]) + np.abs(lon[:] - station[1])
	np.where(diffs <= np.min(diffs))
	x, y = [ v[0] for v in np.where(diffs <= np.min(diffs))]
	return (x, y) if 0 < x < lat.shape[0] and 0 < y < lat.shape[1] else (None, None)

def cut_positions(filename, blurred, *positions):
	blurred = int(blurred)
	pos = eval("".join(positions))
	root = nc.open(filename)[0]
	lat = nc.getvar(root, 'lat')
	lon = nc.getvar(root, 'lon')
	data = nc.getvar(root, 'data')
	root_cut = nc.open('cut_positions.' + filename)[0]
	nc.getdim(root_cut, 'time', None)
	nc.getdim(root_cut, 'yc_cut', len(pos))
	nc.getdim(root_cut, 'xc_cut', 2)
	time = root_cut.getvar('time', 'i4', ('time',))
	slots = root_cut.getvar('slots', 'u1', ('time',))
	lat_cut = root_cut.getvar('lat', 'f4', ('yc_cut','xc_cut',),4)
	lon_cut = root_cut.getvar('lon', 'f4', ('yc_cut','xc_cut',),4)
	data_cut = root_cut.getvar('data', 'f4', ('time', 'yc_cut', 'xc_cut',),4)
	time[:] = nc.getvar(root, 'time')[:]
	slots[:] = nc.getvar(root, 'slots')[:]
	ix = 0
	for i in range(len(pos)):
		show("\rCutting data: processing position %d / %d " % (i+1, len(pos)))
		x, y = statistical_search_position(pos[i], lat, lon)
		if x and y:
			lat_cut[ix,0] = lat[x,y]
			lon_cut[ix,0] = lon[x,y]
			data_cut[:,ix,0] = np.apply_over_axes(np.mean, data[:,x-blurred:x+blurred+1,y-blurred:y+blurred+1], axes=[1,2]) if blurred > 0 else data[:,x,y]
			lat_cut[ix,1], lon_cut[ix,1], data_cut[:,ix,1] = lat_cut[ix,0], lon_cut[ix,0], data_cut[:,ix,0]
			ix += 1
	nc.close(root)
	nc.close(root_cut)

def stations(filename,  stations=[]):

	s = {'Lujan':[-34.5880556, -59.0627778],'Anguil':[-36.541704, -63.990947],'Parana':[-31.84894, -60.536117],'Balcarce':[-37.763199, -58.297519],'Pergamino':[-33.944332, -60.568668],'MarcosJuarez':[-32.568348, -62.082349],'Azul':[-36.766174, -59.881312],'Villegas':[-34.8696, -62.7790],'Barrow':[-38.184, -60.129],'Ceilap':[-34.567, -58.5],'Concepcion':[-32.483, -58.233]}

	positions = []
	for i in stations:
		print i
		print s[i]
		positions.append(str(s[i]))
	blurred = 0
	print positions
	cut_positions(filename, blurred, positions)
	
