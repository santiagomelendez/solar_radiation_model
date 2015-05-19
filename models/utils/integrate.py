#!/usr/bin/env python

import sys
sys.path.append(".")
from netcdf import netcdf as nc
from datetime import datetime
from models.helpers import to_datetime, short
from glob import glob
import numpy as np
from itertools import groupby
import os


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


def separate_data_in_slots(files, slots, window_size):
    dh = lambda filename: decimalhour(to_datetime(filename))
    data_in_slots = map(lambda slot:
                        filter(lambda filename:
                               0 < dh(filename) - slot < window_size + 10./60.,
                               files), slots)
    return data_in_slots


def integral(files, output_filename):
    initial_time = to_datetime(files[0])
    time = map(lambda filename: (to_datetime(filename) -
                                 initial_time).total_seconds(), files)
    root, is_new = nc.open(files)
    radiation = nc.getvar(root, 'globalradiation')
    with nc.loader(output_filename) as integral_root:
        dims_names = list(reversed(radiation.dimensions.keys()))
        dims_values = list(reversed(radiation.dimensions.values()))
        create_dims = (lambda name, dimension:
                       integral_root.create_dimension(name, len(dimension)))
        (map(lambda name, dimension: create_dims(name, dimension),
             dims_names, dims_values))
        integral = (nc.getvar(integral_root, 'integral', vtype='f4',
                              dimensions=tuple(dims_names)))
        integral[:] = np.trapz(radiation[:], time, axis=0)*10**-6


decimalhour = lambda t: t.hour + t.minute/60. + t.second/3600.


def create_output_path(prefix, band, month, window, day, hour):
    if not os.path.exists('products/integral'):
        os.makedirs('products/integral')
    if window == '24':
        output_path = 'products/integral/int.%s.%s.D.%s.%s.nc' % (
            prefix, month, day, band)
    else:
        output_path = 'products/integral/int.%s.%s.H%s.%s.%s.%s.nc' % (
            prefix, month, window, day, hour, band)
    return output_path


def int_by_day(files, window_size):
    """calculate the integral of the files in the same temporal windows for
    different days. """
    julian_day = lambda filename: to_datetime(filename).timetuple()[7]
    prefix = short(files[0], 0, 2)
    band = short(files[0], 4, 5)
    if len(files) > 1:
        for dj, files in groupby(files, julian_day):
            files_to_integrate = list(files)
            month = str(to_datetime(files_to_integrate[0]).month).zfill(2)
            hour = to_datetime(files_to_integrate[0]).hour
            output_filename = create_output_path(prefix, band, month,
                                                 str(window_size),
                                                 str(dj).zfill(3), hour)
            integral(files_to_integrate, output_filename)


def integrate(begin, end, window_size_in_hours):
    path = 'products/estimated/*.nc'
    files = glob(path)
    files.sort()
    selected_files = select_files_range(begin, end, files)
    slots = range(to_datetime(selected_files[0]).hour, 24,
                  window_size_in_hours)
    data_slots = separate_data_in_slots(selected_files, slots,
                                        window_size_in_hours)
    map(lambda f: int_by_day(f, window_size_in_hours), data_slots)


if __name__ == '__main__':
    to_date = lambda s: datetime.strptime(s, "%Y-%m-%d").date()
    begin = to_date(sys.argv[1])
    end = to_date(sys.argv[2])
    window = int(sys.argv[3])
    integrate(begin, end, window)

'#example: python models/utils/integrate.py 2015-02-16 2015-02-25 24'
'#create the diary integral of the days 16 to 25'
