#!/usr/bin/env python2.7

from libs.file import netcdf as nc, clone
from datetime import datetime
import numpy as np
from libs.geometry import jaen as geo
import glob

def getimagedata(filename):
	imagedata = filename.split('/')[-1].split('goes13.')[1].split('.BAND_01')[0]
	return datetime.strptime(imagedata, '%Y.%j.%H%M%S')

def filter(image, threshold):
	image, _ = nc.open(image)
	time = datetime.utcfromtimestamp(nc.getvar(image, 'time')) 
	lat = nc.getvar(image, 'lat')
	lon = nc.getvar(image, 'lon')
	alpha = geo.getsolarelevationmatrix(time, 0.0, lat, lon)
	condition= (np.min(alpha) >= threshold)
	return condition 

def createdict(pattern):
	files = glob.glob(pattern)
	months = { month: { day:[] for day in range(1,32) } for month in range(1,13) }
	for file in files:
		dt = getimagedata(file) 
		dh = dt.hour + dt.minute/60.0
		alpha_condition = filter(file)
		if alpha_condition:
			months[dt.month][dt.day].append(file) 	
	return months

def package(pattern, month, days, filename):
	images = createdict(pattern)
	pattern_list =[]
	for d in days:
		pattern_list += images[month][d]
	pattern_list.sort()
	root, is_new = nc.open(pattern_list)
	root_clone, is_new  = clone.clone(root, filename)		
	return root_clone
