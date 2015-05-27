#!/usr/bin/env python

import sys
sys.path.append(".")
from netcdf import netcdf as nc
from models.helpers import to_datetime, short
from glob import glob
import os


def radiance(filename):
    prefix = short(filename, 0, 3)
    slot = int(round(decimalhour(to_datetime(filename))*2))
    suffix = short(filename, 4, 6)
    output_filename = create_output_path(prefix, slot, suffix)
    root, is_new = nc.open(filename)
    radiation = nc.getvar(root, 'globalradiation')
    with nc.loader(output_filename) as radiance_root:
        dims_names = list(reversed(radiation.dimensions.keys()))
        dims_values = list(reversed(radiation.dimensions.values()))
        create_dims = (lambda name, dimension:
                       radiance_root.create_dimension(name, len(dimension)))
        (map(lambda name, dimension: create_dims(name, dimension),
             dims_names, dims_values))
        radiance = (nc.getvar(radiance_root, 'radiance', vtype='f4',
                              dimensions=tuple(dims_names)))
        radiance[:] = radiation[:]*30.*60.*10**-6


decimalhour = lambda t: t.hour + t.minute/60. + t.second/3600.


def create_output_path(prefix, slot, suffix):
    if not os.path.exists('products/radiance'):
        os.makedirs('products/radiance')
    output_path = 'products/radiance/rad.%s.S%s.%s' % (prefix, slot, suffix)
    return output_path


def workwith(path='products/estimated/*.nc'):
    files = glob(path)
    map(radiance, files)


if __name__ == '__main__':
    workwith()
