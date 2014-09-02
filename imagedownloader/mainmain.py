#!/usr/bin/env python2.7

s = {'Lujan':[-34.5880556, -59.0627778],'Anguil':[-36.541704, -63.990947],'Parana':[-31.84894, -60.536117],'Balcarce':[-37.763199, -58.297519],'Pergamino':[-33.944332, -60.568668],'MarcosJuarez':[-32.568348, -62.082349],'Azul':[-36.766174, -59.881312],'Villegas':[-34.8696, -62.7790],'Barrow':[-38.184, -60.129],'Ceilap':[-34.567, -58.5],'Concepcion':[-32.483, -58.233]}

names = ['Anguil', 'Azul', 'Barrow', 'Concepcion', 'Lujan', 'MarcosJuarez', 'Parana', 'Villegas']
pos = [ s[n] for n in names ]
print pos
from stations.importer import *
from libs.file.toolbox import *
from heliosat.main import *

#cut_positions('clone_M12.nc', 0, pos)

#workwith('2013', '12', '/home/santiago/GERSOLAR/git/solar_radiation_model/imagedownloader/cut_positions.clone_M12.nc')

import_measurement('2013', '12', '/home/santiago/GERSOLAR/git/solar_radiation_model/imagedownloader/cut_positions.clone_M12.nc', names)


from libs.statistics import error
root, _ = nc.open('cut_positions.clone_M12.nc')

error.dailyerrors(root, names)			# Error relativos respecto a la integral diaria

