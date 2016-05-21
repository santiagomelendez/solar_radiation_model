from datetime import datetime
import numpy as np
from netcdf import netcdf as nc
import logging
from linketurbidity import instrument as linke
from noaadem import instrument as dem
from helpers import short
import os


class Cache(object):

    def __init__(self, filenames, tile_cut={}, read_only=False):
        self._attrs = {}
        self.filenames = filenames
        self.tile_cut = tile_cut
        self.root = nc.tailor(filenames, dimensions=tile_cut,
                              read_only=read_only)

    def __getattr__(self, name):
        if name not in self._attrs.keys():
            self.load(name)
        return self._attrs[name]

    def getvar(self, var_name):
        return nc.getvar(self.root, var_name)

    def load(self, name):
        var_name = name[4:] if name[0:4] == 'ref_' else name
        key = 'ref_{:s}'.format(var_name)
        if key not in self._attrs.keys():
            var = self.getvar(var_name)
            self._attrs[key] = var
        else:
            var = self._attrs
        self._attrs[var_name] = var[:]

    def dump(self):
        for k in self._attrs.keys():
            self._attrs.pop(k, None)
        nc.close(self.root)


class StaticCache(Cache):

    @classmethod
    def project_dem(cls, root, lat, lon):
        logging.info("Projecting DEM's map... ")
        dem_var = nc.getvar(root, 'dem', 'f4', source=lon)
        dem_var[:] = dem.obtain(lat[0], lon[0])

    @classmethod
    def project_linke(cls, root, lat, lon):
        logging.info("Projecting Linke's turbidity index... ")
        dts = map(lambda m: datetime(2014, m, 15), range(1, 13))
        linkes = map(lambda dt: linke.obtain(dt, compressed=True), dts)
        linkes = map(lambda l: linke.transform_data(l, lat[0],
                                                    lon[0]), linkes)
        linkes = np.vstack([[linkes]])
        nc.getdim(root, 'months', 12)
        linke_var = nc.getvar(root, 'linke', 'f4', ('months', 'yc', 'xc'))
        # The linkes / 20. uncompress the linke coefficients and save them as
        # floats.
        linke_var[:] = linkes / 20.

    @classmethod
    def construct(cls, static_file, ref_filename):
        # At first it should have: lat, lon, dem, linke
        logging.info("This is the first execution from the deployment... ")
        with nc.loader(ref_filename) as root_ref:
            with nc.loader(static_file) as root:
                lat = nc.getvar(root_ref, 'lat')
                lon = nc.getvar(root_ref, 'lon')
                nc.getvar(root, 'lat', source=lat)
                nc.getvar(root, 'lon', source=lon)
                cls.project_dem(root, lat, lon)
                cls.project_linke(root, lat, lon)

    def __init__(self, static_filename, data_filenames, tile_cut):
        if not os.path.exists(static_filename):
            StaticCache.construct(static_filename,
                                  data_filenames[0])
        super(StaticCache, self).__init__(static_filename, tile_cut)

    @property
    def dem(self):
        if not hasattr(self, '_cached_dem'):
            self._cached_dem = nc.getvar(self.root, 'dem')[:]
        return self._cached_dem

    @property
    def linke(self):
        if not hasattr(self, '_linke'):
            self._linke = nc.getvar(self.root, 'linke')[:]
        return self._linke


class OutputCache(Cache):

    def __init__(self, product, tile_cut, ref_filenames):
        super(OutputCache, self).__init__(ref_filenames,
                                          tile_cut)
        self.product = product
        self.initialize_variables(self.filenames)

    def create_1px_dimensions(self, root):
        nc.getdim(root, 'xc_k', 1)
        nc.getdim(root, 'yc_k', 1)
        nc.getdim(root, 'time', 1)

    def initialize_path(self, filenames, images):
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        self.output = Cache(map(self.get_output_file, self.filenames),
                            tile_cut=self.tile_cut)
        self.root = self.output.root
        if len(filenames) == 1:
            map(self.create_1px_dimensions, [self.root.root])
        else:
            map(self.create_1px_dimensions, self.root.roots)
        self.root.getvar('time', source=images.getvar('time'))
        self.root.getvar('cloudindex', 'f4', source=images.getvar('data'))
        self.root.getvar('globalradiation', 'f4', source=images.getvar('data'))
        self.root.getvar('data_mix', 'f4', source=images.getvar('data'))

    def initialize_variables(self, filenames):
        filenames = ([filenames] if isinstance(filenames, str)
                     else filenames)
        self.path = '/'.join(filenames[0].split('/')[0:-1])
        self.output_path = self.product
        with nc.loader(self.filenames, dimensions=self.tile_cut) as images:
            if self.output_path:
                self.initialize_path(filenames, images)
            else:
                data_shape = images.getvar('data').shape
                self.time = np.zeros(images.getvar('time').shape)
                self.ref_cloudindex = np.zeros(data_shape)
                self.cloudindex = self.ref_cloudindex
                self.ref_globalradiation = np.zeros(data_shape)
                self.globalradiation = self.ref_globalradiation
                self.ref_data_mix = np.zeros(data_shape)
                self.data_mix = self.ref_data_mix

    def get_output_file(self, filename):
        return '{:s}/{:s}'.format(self.output_path,
                                  short(filename, None, None))


class memoize(object):

    def __init__(self, function):
        self.function = function
        self.memoized = {}

    def __call__(self, *args):
        try:
            return self.memoized[args]
        except KeyError:
            self.memoized[args] = self.function(*args)
        return self.memoized[args]
