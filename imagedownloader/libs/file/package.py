#!/usr/bin/env python2.7

import sys
sys.path.append('.')
from libs.file import netcdf as nc, clone
from datetime import datetime
import numpy as np
from libs.console import show, say
from libs.geometry import jaen as geo
import glob
from itertools import chain, groupby, dropwhile
from functools import partial
import re

def getimagedata(filename):
	imagedata = re.findall(r'goes13\.(.*)\.(.*)\.(.*)\.BAND_01\.nc', filename)
	all_digits = lambda t: all([i.isdigit() for i in t])	
	valid_datetime = lambda d: len(d) > 0 and all_digits(d[0])
	return (datetime.strptime('%s.%s.%s' % imagedata[0], '%Y.%j.%H%M%S') 
			  if valid_datetime(imagedata) else None)
			   
def solar_elevation_condition(lat, lon, threshold, time):
	alpha = geo.getsolarelevationmatrix(time, 0.0, lat, lon)
	condition= (np.min(alpha) >= threshold)
	return condition 


def pack(pattern_list, filename):
	pattern_list.sort()
	root, is_new = nc.open(pattern_list)
	root_clone, _  = clone.clone(root, filename)		
	return root_clone

def filter_by_month(files, month):
	valid_date = lambda t, month: t.month == month and t.hour > 8 and t.hour < 22
	month_images = filter(lambda file, m=month: valid_date(getimagedata(file), m), files)	
	month_images.sort()
	return month_images

def divide_by_day(month_images):
	month_images.sort()
	days_list = []
	days_key = []
	for day, files in groupby(month_images, lambda f: getimagedata(f).day):
		days_list.append(list(files))
		days_key.append(day)       
	return days_list, days_key

def select_day_images(lat, lon, day_list):
    alpha = partial(solar_elevation_condition, lat, lon, 10.0)
    is_night = lambda image: not alpha(getimagedata(image))
    drop_night_from_left = lambda l: list(dropwhile(is_night, l)) 
    day = drop_night_from_left(day_list) 
    day.reverse()
    day = drop_night_from_left(day) 
    day.reverse()
    return day

def select_images(pattern, month):
	from multiprocessing import Pool
	files = glob.glob(pattern)
	image, _ = nc.open(files[0])
	lat = nc.getvar(image, 'lat')
	lon = nc.getvar(image, 'lon')
	nc.close(image)
	month_images = filter_by_month(files, month)
	days_images, days_keys = divide_by_day(month_images)
	day_selection = partial(select_day_images, lat, lon)
	pool = Pool()
	data = pool.map(day_selection, days_images)
	data = list(chain(*data))
	return data

if __name__=="__main__":
	pattern = sys.argv[1]
	month = int(sys.argv[2])
	year = sys.argv[3]
	data = select_images(pattern, month)	
	filename = 'M' + str(month) + '_' + year + '.nc'
	package = pack(data[0:30], filename)
	nc.sync(package)
	nc.close(package)
	show("\nReady.\n")
