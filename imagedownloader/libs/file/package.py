#!/usr/bin/env python2.7

import sys
sys.path.append('.')
from libs.file import netcdf as nc, clone
from datetime import datetime
import numpy as np
from libs.console import show, say
from libs.geometry import jaen as geo
import glob
from itertools import chain
import re

def getimagedata(filename):
	imagedata = re.findall(r'goes13\.(.*)\.(.*)\.(.*)\.BAND_01\.nc', filename)
	all_digits = lambda t: all([i.isdigit() for i in t])	
	valid_datetime = lambda d: len(d) > 0 and all_digits(d[0])
	return (datetime.strptime('%s.%s.%s' % imagedata[0], '%Y.%j.%H%M%S') 
			  if valid_datetime(imagedata) else None)
			   
def filter_by_solar_elevation(time, lat, lon, threshold):
	alpha = geo.getsolarelevationmatrix(time, 0.0, lat, lon)
	condition= (np.min(alpha) >= threshold)
	return condition 

def create_dict(pattern, year, month):
	files = glob.glob(pattern)
	files.sort()
	image, _ = nc.open(files[0])
	lat = nc.getvar(image, 'lat')
	lon = nc.getvar(image, 'lon')
	nc.close(image)
	days = {}
	valid_data = lambda t: t and t.month 
	for file in files:
		dt = getimagedata(file) 
		if valid_data(dt):
			alpha_condition = filter_by_solar_elevation(dt, lat, lon, 10)
			if alpha_condition:
				days.setdefault(dt.day, []).append(file) 	
	return days

def pack(satellite, filename, year, month, *day):
	month = create_dict(satellite, year, month)
	pattern = ([month[d].values() for d in day]
					  if len(day) > 0 else month.values())
	pattern = list(chain(*pattern))
	pattern.sort()
	root, is_new = nc.open(pattern)
	root_clone, _  = clone.clone(root, filename)		
	return root_clone, month

if __name__=="__main__":
	functions = {'create_dict' : create_dict,
		'pack' : pack}
	if sys.argv[1] == 'create_dict':
		days = functions[sys.argv[1]](*(sys.argv[2:]))
		dict = (open('libs/file/' + 'M' + sys.argv[4] + '_' 
					+ sys.argv[3] + '_dict' + '.txt', 'w'))
		for d in days.keys():
			files = [image.split('/')[-1] for image in days[d]]
			dict.write(str(d) + ': ' + str(files) + '\n')
		dict.close()
	else:
			functions[sys.argv[1]](*(sys.argv[2:]))
	show("\nReady.\n")

