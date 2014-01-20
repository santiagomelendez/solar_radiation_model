import sys
sys.path.append("../")
from libs.file import netcdf as nc
from libs.console import show
import numpy as np
from datetime import datetime

def rmse(root, index):
	times = [ datetime.utcfromtimestamp(int(t)) for t in nc.getvar(root, 'data_time')[:] ]
	days = [ t.date() for t in times ]
	days.sort()
	days_index = [d.day for d in set(days)]
	days_amount = len(days_index)
	nc.getdim(root, 'diarying', days_amount)
	nc.sync(root)
	measurements = nc.getvar(root, 'measurements')
	estimated = nc.getvar(root, 'globalradiation')
	error_diff = nc.getvar(root, 'errordiff', 'f4', ('timing','northing_cut','easting_cut',), 4)
	error = nc.getvar(root, 'error', 'f4', ('timing','northing_cut','easting_cut',), 4)
	diary_error = nc.getvar(root, 'diaryerror', 'f4', ('diarying', 'northing_cut', 'easting_cut',), 4)
	error_diff[:] = np.zeros(estimated.shape)
	error[:] = np.zeros(estimated.shape)
	diary_error[:] = np.zeros((days_amount, estimated.shape[1], estimated.shape[2]))
	nc.sync(root)
	#the_max = measurements[:].max()
	error_diff[:, index, :] = measurements[:,index,:] - estimated[:,index,:]
	error[:, index, :] = np.abs(error_diff[:, index, :])
	nc.sync(root)
	max_value_in_day = np.zeros([days_amount]) + 1
	for i in range(len(days)):
		d_i = days_index.index(days[i].day)
		max_value_in_day[d_i] = measurements[i,index,0] if max_value_in_day[d_i] < measurements[i,index,0] else max_value_in_day[d_i]
		diary_error[d_i, index,:] += np.array([ error_diff[i,index,0] ** 2,1])
	count = diary_error[:, index, 1]
	count[count == 0] = 1
	diary_error[:, index,0] = np.sqrt(diary_error[:, index, 0] / count)
	diary_error[:, index,1] = diary_error[:, index,0] / max_value_in_day * 100
	show("\rDiary RMS error: %.2f" % (diary_error[:, index, 1]).mean())
	for i in range(len(days)):
		d_i = days_index.index(days[i].day)
		error[i,index,1] = error[i,index,1] / max_value_in_day[d_i] * 100
	result = np.sum(error[:, index, 1] ** 2)
	result = np.sqrt(result / error.shape[0])
	show("Half-hour RMS error: %.2f \n" % result)
	#diary_error[:, index,1] = diary_error[:, index,0]
	nc.sync(root)


def diff(estimated, measured, station):
	return estimated[:,station,0] - measured[:,station,0]

def ghi_mean(measured, station):
	return measured[:,station,0].mean()

def ghi_ratio(measured, station):
	return 100 / ghi_mean(measured, station)

def bias(estimated, measured, station):
	t_diff = diff(estimated, measured, station)
	return t_diff.mean()

def rmse_es(estimated, measured, station):
	t_diff = diff(estimated, measured, station)
	return np.sqrt((t_diff**2).mean())

def mae(estimated, measured, station):
	t_diff = diff(estimated, measured, station)
	return np.absolute(t_diff).mean()

filename = sys.argv[1] if len(sys.argv) == 2 else None
if not filename is None:
	try:
		index = int(sys.argv[2])
		root,n = nc.open(filename)
		rmse(root, index)
		nc.close(root)
	except Exception, e:
		show(e)
