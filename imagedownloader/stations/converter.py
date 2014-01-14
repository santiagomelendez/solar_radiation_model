import datetime
import sys
import numpy as np
import pytz
import csv


def matrix2csv(matrix):
	with open(sys.argv[2], 'w') as fp:
		a = csv.writer(fp, delimiter=',',  quoting=csv.QUOTE_NONNUMERIC)
		a.writerows(matrix)


def datetime_convert_to_iso(year, julianday, hour):
	naive_date = datetime.date(int(year), 1, 1) + datetime.timedelta(int(julianday) -1)
	naive_time = datetime.datetime.strptime((str(hour)).zfill(4), '%H%M')
	naive_time = naive_time.time()
	naive_iso_time = datetime.datetime.combine(naive_date, naive_time)
	return naive_iso_time


def naive_to_aware_time(naivetime, tz ):
	local_time = tz.localize(naivetime)
	utc_time = local_time.astimezone(pytz.utc)
	return utc_time

# a modificar en el futuro para acceder a datos de otras zona
localtz = pytz.timezone('America/Argentina/Buenos_Aires')

matriz = []

x = np.genfromtxt(fname=sys.argv[1],
	delimiter=',',
	dtype = (int, int, int, float))

integration_time = int(sys.argv[3]) * 60
for row in xrange(x.size):
	matriz.append((naive_to_aware_time(datetime_convert_to_iso(x[row][0], x[row][1], x[row][2]),localtz),float("%.2f" % x[row][3])/integration_time ))
matrix2csv(matriz)

#python converter.py 2011.csv 2011UTCmean 10