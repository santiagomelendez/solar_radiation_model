import datetime
import sys
import numpy as np


x = np.genfromtxt(fname=sys.argv[1],
	delimiter=',',
	dtype = (int, int, int, float))

def time_convert_to_posix(year, julianday, hour):
	fecha = datetime.date(int(year), 1, 1) + datetime.timedelta(int(julianday) -1)
	horario = datetime.datetime.strptime((str(hour)).zfill(4), '%H%M')
	horario = horario.time()
	tiempo_posix = datetime.datetime.combine(fecha, horario)
	return tiempo_posix

for row in xrange(x.size):
	timestamp = time_convert_to_posix(x[row][0], x[row][1], x[row][2])
	print timestamp, x[row][3]