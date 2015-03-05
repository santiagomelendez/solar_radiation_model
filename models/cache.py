from datetime import datetime
import numpy as np
from netcdf import netcdf as nc
from helpers import to_datetime
from helpers import show
from linketurbidity import instrument as linke
from noaadem import instrument as dem


class StaticCache(object):

    def __init__(self, filenames):
        # At first it should have: lat, lon, dem, linke
        self.root, is_new = nc.open('static.nc')
        if is_new:
            show("This is the first execution from the deployment... ")
            with nc.loader(filenames[0]) as root_ref:
                self.lat = nc.getvar(root_ref, 'lat')
                self.lon = nc.getvar(root_ref, 'lon')
                nc.getvar(self.root, 'lat', source=self.lat)
                nc.getvar(self.root, 'lon', source=self.lon)
                self.project_dem()
                self.project_linke()
                nc.sync(self.root)
            show("-----------------------\n")

    def project_dem(self):
        show("Projecting DEM's map... ")
        dem_var = nc.getvar(self.root, 'dem', 'f4', source=self.lon)
        dem_var[:] = dem.obtain(self.lat[0], self.lon[0])

    def project_linke(self):
        show("Projecting Linke's turbidity index... ")
        dts = map(lambda m: datetime(2014, m, 15), range(1, 13))
        linkes = map(lambda dt: linke.obtain(dt, compressed=False), dts)
        linkes = map(lambda l: linke.transform_data(l, self.lat[0],
                                                    self.lon[0]), linkes)
        linkes = np.vstack([[linkes]])
        nc.getdim(self.root, 'months', 12)
        linke_var = nc.getvar(self.root, 'linke', 'f4', ('months', 'yc', 'xc'))
        linke_var[:] = linkes


class Loader(object):

    def __init__(self, filenames):
        self.filenames = filenames
        self.root = nc.open(filenames)[0]
        self.static = StaticCache(filenames)
        self.static_cached = self.static.root
        self._attrs = {}

    @property
    def dem(self):
        if not hasattr(self, '_cached_dem'):
            self._cached_dem = nc.getvar(self.static_cached, 'dem')
        return self._cached_dem

    @property
    def linke(self):
        if not hasattr(self, '_cached_linke'):
            self._linke = nc.getvar(self.static_cached, 'linke')
            self._cached_linke = np.vstack([
                map(lambda dt: self._linke[0, dt.month - 1],
                    map(to_datetime, self.filenames))])
        return self._cached_linke

    @property
    def calibrated_data(self):
        if not hasattr(self, '_cached_calibrated_data'):
            row_data = self.data[:]
            counts_shift = self.counts_shift[:]
            space_measurement = self.space_measurement[:]
            prelaunch = self.prelaunch_0[:]
            postlaunch = self.postlaunch[:]
            # INFO: Without the postlaunch coefficient the RMSE go to 15%
            normalized_data = (np.float32(row_data) / counts_shift -
                               space_measurement)
            self._cached_calibrated_data = (normalized_data
                                            * postlaunch
                                            * prelaunch)
        return self._cached_calibrated_data

    def __getattr__(self, name):
        if name not in self._attrs.keys():
            self._attrs[name] = nc.getvar(self.root, name)
        return self._attrs[name]
