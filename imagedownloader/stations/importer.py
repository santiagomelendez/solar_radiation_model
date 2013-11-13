import sys
sys.path.append("../")
sys.path.append(".")
from datetime import datetime, timedelta
from xlrd import open_workbook, XL_CELL_EMPTY, XL_CELL_BLANK
import csv
from libs.file import netcdf as nc
from libs.geometry import jaen as geo
from libs.console import *
import numpy as np
from libs.statistics import error

def rows2csv(rows, filename):
	with open(filename, 'wb') as csvfile:
		spamwriter = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
		for i in range(len(rows)):
			spamwriter.writerow(rows[i])

def mvolt_to_watt(mvolt):
	mvolt_per_kwatts_m2 = 8.48
	return (mvolt / mvolt_per_kwatts_m2) * 1000

def rows2slots(rows, image_per_hour):
	resumed_slots = []
	old_slot = geo.getslots(rows[0][0],image_per_hour)
	seconds_in_slot = (rows[1][0] - rows[0][0]).total_seconds()
	mvolt = 0
	rows_by_slot = 0
	dt = rows[0][0]
	for r in rows:
		slot = geo.getslots(dt,image_per_hour)
		if not old_slot is slot:
			resumed_slots.append([slot, [dt, mvolt_to_watt((mvolt/rows_by_slot)/seconds_in_slot), rows_by_slot]])
			old_slot, rows_by_slot, mvolt = slot, 0, 0
		mvolt += r[1]
		rows_by_slot += 1
		dt = r[0]
	if not old_slot is slot:
		resumed_slots.append([slot, [dt, mvolt_to_watt((mvolt/rows_by_slot)/seconds_in_slot), rows_by_slot]])
		old_slot, rows_by_slot, mvolt = slot, 0, 0
	return resumed_slots

def rows2netcdf(rows, filename, index):
	root, is_new = nc.open(filename)
	if not is_new:
		measurements = nc.clonevar(root, 'estimated', 'measurements')
		slots = nc.getvar(root, 'slots')
		times = [ datetime.utcfromtimestamp(int(t)) for t in nc.getvar(root, 'data_time')[:] ]
		instant_radiation = rows2slots(rows,2)
		i_e = 0
		i_m = 0
		while i_e < len(times) and i_m < len(instant_radiation):
			# When date estimated before date measured
			if times[i_e].date() < instant_radiation[i_m][1][0].date():
				i_e += 1
			# When date estimated after date measured
			elif times[i_e].date() > instant_radiation[i_m][1][0].date():
				i_m += 1
			else:
				if slots[i_e] < instant_radiation[i_m][0]:
					# TODO: This should be completed with a 0 error from the estimation.
					measurements[i_e, index,:] = np.array([0, 0])
					show("Detected estimated time without earth measure.")
					i_e += 1
				elif slots[i_e] > instant_radiation[i_m][0]:
					i_m += 1
				else:
					value = instant_radiation[i_m][1][1]
					row_in_slot = instant_radiation[i_m][1][2]
					measurements[i_e, index,:] = np.array([value, value])
					i_e += 1
					i_m += 1
		while i_e < len(times):
			# TODO: This should be completed with a 0 error from the estimation.
			measurements[i_e, index,:] = np.array([0, 0])
			show("Detected estimated time without earth measure.")
			i_e += 1
		error.rmse(root, index)
		nc.close(root)

def get_val(sh,x,y):
	try:
		cell = sh.cell(y,x)
		result = None if cell.ctype in [XL_CELL_EMPTY, XL_CELL_BLANK] else cell.value
	except:
		result = None
	return result

def to_datetime(year_or_timestamp, julian=None, hour="00", minute="00", second="00", utc_hour=0, utc_minute=0):
	if year_or_timestamp.__class__ is str and julian is None:
		tzsplit = year_or_timestamp.split("+")
		timestamp = datetime.strptime(tzsplit[0],"%Y-%m-%d %H:%M:%S")
		if len(tzsplit) > 1:
			utc_hour, utc_minute = int(tzsplit[1][:2]) if utc_hour is 0 else utc_hour, int(tzsplit[1][4:])
	else:
		# This is for Julian days
		s = "%i-%i %s:%s:%s" % (int(year_or_timestamp),int(julian),hour,minute,second)
		timestamp = datetime.strptime(s,"%Y-%j %H:%M:%S")
	delta =  timedelta(hours = abs(utc_hour), minutes=abs(utc_minute))
	timestamp = (timestamp + delta if utc_hour < 0 else timestamp - delta)
	return timestamp

def from_xls(input_filename, utc_diff):
	i_sheet = int(sys.argv[5])
	x_year = int(sys.argv[6])
	x_julian = int(sys.argv[7])
	x_timestamp = int(sys.argv[8])
	x_value = int(sys.argv[9])
	y_from = int(sys.argv[10])
	wb = open_workbook(input_filename)
	sh = wb.sheets()[i_sheet]
	rows = []
	y = y_from
	year, julian, time = get_val(sh,x_year,y), get_val(sh,x_julian,y), get_val(sh,x_timestamp,y)
	while not(year is None and julian is None and time is None):
		time = str(int(time)).zfill(4)
		timestamp = to_datetime(year, julian, time[0:2],time[2:4], utc_hour=utc_diff, utc_minute=0)
		data = get_val(sh,x_value,y)
		if not data is None:
			try:
				rows.append((timestamp, float(data)))
			except Exception, e:
				print e
		y += 1
		year, julian, time = get_val(sh,x_year,y), get_val(sh,x_julian,y), get_val(sh,x_timestamp,y)
	return rows

def from_csv(input_filename, utc_diff):
	timestamp_col = int(sys.argv[5])
	channel = int(sys.argv[6])
	skip_rows = int(sys.argv[7])
	rows = np.genfromtxt(input_filename,
		delimiter = ',',
		skiprows= skip_rows,
		usecols=[0, channel],
		converters = {0: lambda s: to_datetime(s[1:-1], utc_hour=int(utc_diff)), channel: lambda s: float(s)})
	return rows

importer = {}
importer["xls"] = from_xls
importer["csv"] = from_csv

output_filename = sys.argv[1]
output_index = int(sys.argv[2]) if output_filename.split(".")[-1] == "nc" else None
input_filename = sys.argv[3]
utc_diff = int(sys.argv[4])
rows = importer[input_filename.split(".")[-1]](input_filename, utc_diff)
if output_index is None:
	rows2csv(rows, output_filename)
else:
	rows2netcdf(rows,output_filename,output_index)

#python2.7 importer.py cut_positions.pkg.goes13.all.BAND_01.nc 0 mayo2011.xls -3 1 1 2 3 9 10
#python2.7 importer.py cut_positions.pkg.goes13.all.BAND_01.nc 0 2011UTC.csv 0 0 1 3

