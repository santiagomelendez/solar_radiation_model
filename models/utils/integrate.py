#!/usr/bin/env python

import sys
sys.path.append(".")
from netcdf import netcdf as nc
from datetime import datetime
from models.helpers import to_datetime
from glob import glob
import numpy as np
from itertools import groupby
import os


def pdb():
    import pdb
    pdb.set_trace()


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


decimalhour = lambda t: t.hour + t.minute/60. + t.second/3600.


def separate_data_in_slots(files, slots, window_size):
    dh = lambda filename: decimalhour(to_datetime(filename))
    data_in_slots = map(lambda slot:
                        filter(lambda filename:
                               0 < dh(filename) - slot < window_size + 10./60.,
                               files), slots)
    return data_in_slots


def integrate(begin, end, window_size_in_hours):
    path = 'products/estimated/*.nc'
    files = glob(path)
    files.sort()
    selected_files = select_files_range(begin, end, files)
    slots = lambda filename: int(round(decimalhour(to_datetime(filename))))
    for slot, files in groupby(selected_files, slots):
        #print slot, list(files)
        print slot, os.path.commonprefix(list(files))
        # filename = 'int.%s'
        # output_filename = 'products/integrals/%s' % filename
        # print output_filename


if __name__ == '__main__':
    to_date = lambda s: datetime.strptime(s, "%Y-%m-%d").date()
    begin = to_date(sys.argv[1])
    end = to_date(sys.argv[2])
    window = int(sys.argv[3])
    integrate(begin, end, window)
