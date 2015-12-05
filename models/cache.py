from datetime import datetime
import numpy as np
from netcdf import netcdf as nc
import logging
from linketurbidity import instrument as linke
from noaadem import instrument as dem


class Cache(object):

    def __init__(self):
        self._attrs = {}

    def __getattr__(self, name):
        if name not in self._attrs.keys():
            self.load(name)
        return self._attrs[name]

    def getvar(self, var_name):
        return nc.getvar(self.root, var_name)

    def load(self, name):
        var_name = name[4:] if name[0:4] == 'ref_' else name
        if 'ref_%s' % var_name not in self._attrs.keys():
            var = self.getvar(var_name)
            self._attrs['ref_%s' % var_name] = var
        else:
            var = self._attrs
        self._attrs[var_name] = var[:]

    def dump(self):
        for k in self._attrs.keys():
            self._attrs.pop(k, None)
        nc.close(self.root)


class StaticCache(Cache):

    def __init__(self, algorithm):
        ref_filename = algorithm.filenames[0]
        tile_cut = algorithm.config['tile_cut']
        # At first it should have: lat, lon, dem, linke
        self.root, is_new = nc.open(algorithm.config['static_file'])
        if is_new:
            logging.info("This is the first execution from the deployment... ")
            with nc.loader(ref_filename) as root_ref:
                self.lat = nc.getvar(root_ref, 'lat')
                self.lon = nc.getvar(root_ref, 'lon')
                nc.getvar(self.root, 'lat', source=self.lat)
                nc.getvar(self.root, 'lon', source=self.lon)
                self.project_dem()
                self.project_linke()
                nc.sync(self.root)
        self.root = nc.tailor(self.root, dimensions=tile_cut)
        self.lat = nc.getvar(self.root, 'lat')[:]
        self.lon = nc.getvar(self.root, 'lon')[:]

    def project_dem(self):
        logging.info("Projecting DEM's map... ")
        dem_var = nc.getvar(self.root, 'dem', 'f4', source=self.lon)
        dem_var[:] = dem.obtain(self.lat[0], self.lon[0])

    def project_linke(self):
        logging.info("Projecting Linke's turbidity index... ")
        dts = map(lambda m: datetime(2014, m, 15), range(1, 13))
        linkes = map(lambda dt: linke.obtain(dt, compressed=True), dts)
        linkes = map(lambda l: linke.transform_data(l, self.lat[0],
                                                    self.lon[0]), linkes)
        linkes = np.vstack([[linkes]])
        nc.getdim(self.root, 'months', 12)
        linke_var = nc.getvar(self.root, 'linke', 'f4', ('months', 'yc', 'xc'))
        # The linkes / 20. uncompress the linke coefficients and save them as
        # floats.
        linke_var[:] = linkes / 20.

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


class Loader(Cache):

    def __init__(self, filenames, tile_cut={}, read_only=False):
        super(Loader, self).__init__()
        self.root = nc.tailor(filenames, dimensions=tile_cut,
                              read_only=read_only)


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
