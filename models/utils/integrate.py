#!/usr/bin/env python

import sys
sys.path.append(".")
from netcdf import netcdf as nc
from datetime import datetime, timedelta
from models.helpers import to_datetime
from glob import glob
import numpy as np

def select_files_range(begin, end, files):
    """ if begin and end are the same date, select the files of 
    the same day """ 
    files.sort()
    dates = map(lambda filename: to_datetime(filename).date(), files)
    dates_last_index = (lambda value: dates.index(value) + 
        dates.count(value)) 
    begin_index = dates.index(begin)
    end_index = dates_last_index(end)
    return files[begin_index:end_index]

def integral(files, output_filename):
    initial_time = to_datetime(files[0])
    time = (map(lambda filename: (to_datetime(filename) - 
         initial_time).total_seconds(), files))
    root, is_new = nc.open(files)
    radiation = nc.getvar(root, 'globalradiation')
    with nc.loader(output_filename) as integral_root:
        dims_names = list(reversed(root.roots[0].dimensions.keys()))
        dims_values = list(reversed(root.roots[0].dimensions.values()))
        create_dims = (lambda name, dimension: 
            integral_root.create_dimension(name, len(dimension[0])))
        (map(lambda name, dimension: create_dims(name, dimension), 
            dims_names, dims_values))
        integral = (nc.getvar(integral_root, 'integral', vtype='f4', 
            dimensions=tuple(dims_names)))
        integral[:] = np.trapz(radiation[:], time, axis=0)*10**-6






