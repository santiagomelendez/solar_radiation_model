#!/usr/bin/env python

import sys
sys.path.append(".")
from netcdf import netcdf as nc
from models.helpers import to_datetime, short
from glob import glob
import os


def radiance(filename):
    radiance_filename = generate_radiance_filename(filename)
    root, is_new = nc.open(filename)
    radiation = nc.getvar(root, 'globalradiation')
    with nc.loader(radiance_filename) as radiance_root:
	radiance = nc.getvar(radiance_root, 'radiance', source=radiation)
        radiance[:] = radiation[:]*30.*60.*10**-6


decimalhour = lambda t: t.hour + t.minute/60. + t.second/3600.


def generate_radiance_filename(filename):
    prefix = short(filename, 0, 3)
    slot = int(round(decimalhour(to_datetime(filename))*2))
    suffix = short(filename, 4, 6)
    if not os.path.exists('products/radiance'):
        os.makedirs('products/radiance')
    output_filename = 'products/radiance/rad.%s.S%s.%s' % (prefix, slot, suffix)
    return output_filename


def workwith(path='products/estimated/*.nc'):
    files = glob(path)
    map(radiance, files)


if __name__ == '__main__':
    workwith()
