#!/usr/bin/env python

import numpy as np
from datetime import datetime, timedelta
import glob
import os
from netcdf import netcdf as nc
from cache import Loader
from helpers import to_datetime, short, show
import stats

from core import cuda_can_help
if cuda_can_help:
    import gpu as geo
else:
    import cpu as geo
# import processgroundstations as pgs

SAT_LON = -75.113  # -75.3305 # longitude of sub-satellite point in degrees
IMAGE_PER_HOUR = 2
GOES_OBSERVED_ALBEDO_CALIBRATION = 1.89544 * (10 ** (-3))


class Heliosat2(object):

    def __init__(self, filenames):
        self.filenames = filenames
        self.i0met = np.pi / GOES_OBSERVED_ALBEDO_CALIBRATION
        self.cache = TemporalCache(self)

    def process_temporalcache(self, loader, cache):
        lat = loader.lat
        lon = loader.lon
        data = loader.data
        time = loader.time[:]
        shape = list(time.shape)
        shape.append(1)
        time = time.reshape(tuple(shape))
        nc.getdim(cache, 'xc_k', 1)
        nc.getdim(cache, 'yc_k', 1)
        nc.getdim(cache, 'time', 1)
        slots = cache.getvar('slots', 'i1', ('time', 'yc_k', 'xc_k'))
        slots[:] = geo.getslots(time, IMAGE_PER_HOUR)
        nc.sync(cache)
        declination = cache.getvar('declination', source=slots)
        solarangle = cache.getvar('solarangle', 'f4', source=data)
        solarelevation = cache.getvar('solarelevation', source=solarangle)
        excentricity = cache.getvar('excentricity', source=slots)
        show("Calculaing time related paramenters... ")
        result = geo.obtain_gamma_params(time, lat[0], lon[0])
        declination[:], solarangle[:], solarelevation[:], excentricity[:] = result
        linketurbidity = loader.linke[:]
        terrain = loader.dem[:]
        show("Calculating beam, diffuse and global irradiance... ")
        # The average extraterrestrial irradiance is 1367.0 Watts/meter^2
        bc = geo.getbeamirradiance(1367.0, excentricity[:], solarangle[:],
                                   solarelevation[:], linketurbidity, terrain)
        dc = geo.getdiffuseirradiance(1367.0, excentricity[:], solarelevation[:],
                                      linketurbidity)
        gc = geo.getglobalirradiance(bc, dc)
        v_gc = nc.getvar(cache, 'gc', source=data)
        v_gc[:] = gc
        v_gc, v_dc = None, None
        show("Calculating the satellital parameters... ")
        satellitalzenithangle = geo.getsatellitalzenithangle(lat[:], lon[:],
                                                             SAT_LON)
        show("Calculating atmospheric irradiance... ")
        atmosphericradiance = geo.getatmosphericradiance(1367.0,
                                                         self.i0met,
                                                         dc,
                                                         satellitalzenithangle)
        atmosphericalbedo = geo.getalbedo(atmosphericradiance, self.i0met,
                                          excentricity[:],
                                          satellitalzenithangle)
        satellitalelevation = geo.getelevation(satellitalzenithangle)
        v_atmosphericalbedo = nc.getvar(cache, 'atmosphericalbedo',
                                        source=data)
        v_atmosphericalbedo[:] = atmosphericalbedo
        show("Calculating satellital optical path and optical depth... ")
        satellital_opticalpath = geo.getopticalpath(
            geo.getcorrectedelevation(satellitalelevation), terrain, 8434.5)
        satellital_opticaldepth = geo.getopticaldepth(satellital_opticalpath)
        show("Calculating earth-satellite transmitances... ")
        t_sat = geo.gettransmitance(linketurbidity, satellital_opticalpath,
                                    satellital_opticaldepth, satellitalelevation)
        v_sat = nc.getvar(cache, 't_sat', source=lon)
        v_sat[:] = t_sat
        v_atmosphericalbedo, v_sat = None, None
        show("Calculating solar optical path and optical depth... ")
        # The maximum height of the non-transparent atmosphere is at 8434.5 mts
        solar_opticalpath = geo.getopticalpath(
            geo.getcorrectedelevation(solarelevation[:]), terrain, 8434.5)
        solar_opticaldepth = geo.getopticaldepth(solar_opticalpath)
        show("Calculating sun-earth transmitances... ")
        t_earth = geo.gettransmitance(linketurbidity, solar_opticalpath,
                                      solar_opticaldepth, solarelevation[:])
        v_earth = nc.getvar(cache, 't_earth', source=data)
        v_earth[:] = t_earth
        v_earth, v_sat = None, None
        effectivealbedo = geo.geteffectivealbedo(solarangle[:])
        cloudalbedo = geo.getcloudalbedo(effectivealbedo, atmosphericalbedo,
                                         t_earth, t_sat)
        v_albedo = nc.getvar(cache, 'cloudalbedo', source=data)
        v_albedo[:] = cloudalbedo
        nc.sync(cache)
        v_albedo = None


    def process_globalradiation(self, loader, cache):
        excentricity = cache.excentricity[:]
        solarangle = cache.solarangle[:]
        atmosphericalbedo = cache.atmosphericalbedo[:]
        t_earth = cache.t_earth[:]
        t_sat = cache.t_sat[:]
        show("Calculating observed albedo, apparent albedo, effective albedo and "
            "cloud albedo... ")
        observedalbedo = geo.getalbedo(loader.calibrated_data, self.i0met,
                                       excentricity, solarangle)
        apparentalbedo = geo.getapparentalbedo(observedalbedo, atmosphericalbedo,
                                               t_earth, t_sat)
        slots = cache.slots[:]
        declination = cache.declination[:]
        show("Calculating the noon window... ")
        slot_window_in_hours = 4
        # On meteosat are 96 image per day
        image_per_hour = IMAGE_PER_HOUR
        image_per_day = 24 * image_per_hour
        # and 48 image to the noon
        noon_slot = image_per_day / 2
        half_window = image_per_hour * slot_window_in_hours/2
        min_slot = noon_slot - half_window
        max_slot = noon_slot + half_window
        # Create the condition used to work only with the data inside that window
        show("Filtering the data outside the calculated window... ")
        condition = ((slots >= min_slot) & (slots < max_slot))
        # TODO: Meteosat: From 40 to 56 inclusive (the last one is not included)
        condition = np.reshape(condition, condition.shape[0])
        mask1 = loader.calibrated_data[condition] <= (self.i0met / np.pi) * 0.03
        m_apparentalbedo = np.ma.masked_array(apparentalbedo[condition], mask1)
        # To do the nexts steps needs a lot of memory
        show("Calculating the ground reference albedo... ")
        mask2 = m_apparentalbedo < stats.scoreatpercentile(m_apparentalbedo, 5)
        p5_apparentalbedo = np.ma.masked_array(m_apparentalbedo, mask2)
        groundreferencealbedo = geo.getsecondmin(p5_apparentalbedo)
        # Calculate the solar elevation using times, latitudes and omega
        show("Calculating solar elevation... ")
        lat = loader.lat
        r_alphanoon = geo.getsolarelevation(declination, lat[0], 0)
        r_alphanoon = r_alphanoon * 2./3.
        r_alphanoon[r_alphanoon > 40] = 40
        r_alphanoon[r_alphanoon < 15] = 15
        solarelevation = cache.solarelevation[:]
        show("Calculating the apparent albedo second minimum... ")
        groundminimumalbedo = geo.getsecondmin(
            np.ma.masked_array(apparentalbedo[condition],
                               solarelevation[condition] < r_alphanoon[condition]))
        aux_2g0 = 2 * groundreferencealbedo
        aux_05g0 = 0.5 * groundreferencealbedo
        condition_2g0 = groundminimumalbedo > aux_2g0
        condition_05g0 = groundminimumalbedo < aux_05g0
        groundminimumalbedo[condition_2g0] = aux_2g0[condition_2g0]
        groundminimumalbedo[condition_05g0] = aux_05g0[condition_05g0]
        show("Synchronizing with the NetCDF4 file... ")
        cloudalbedo = cache.cloudalbedo
        show("Calculating the cloud index... ")
        cloudindex = geo.getcloudindex(apparentalbedo, groundminimumalbedo,
                                       cloudalbedo[:])
        apparentalbedo = None
        show("Calculating the clear sky... ")
        clearsky = geo.getclearsky(cloudindex)
        show("Calculating the global radiation... ")
        clearskyglobalradiation = nc.getvar(cache.root, 'gc')
        globalradiation = clearsky * clearskyglobalradiation[:]
        f_var = nc.getvar(cache.root, 'globalradiation',
                          source=clearskyglobalradiation)
        show("Saving the global radiation... ")
        f_var[:] = globalradiation
        nc.sync(cache.root)
        f_var = None
        cloudalbedo = None


    def process_validate(self, root):
        from libs.statistics import error
        estimated = nc.getvar(root, 'globalradiation')
        measured = nc.getvar(root, 'measurements')
        stations = [0]
        for s in stations:
            show("==========\n")
            show("Station %i (%i slots)" % (s,  measured[:, s, 0].size))
            show("----------")
            show("mean (measured):\t", error.ghi_mean(measured, s))
            show("mean (estimated):\t", estimated[:, s, 0].mean())
            ghi_ratio = error.ghi_ratio(measured, s)
            bias = error.bias(estimated, measured, s)
            show("BIAS:\t%.5f\t( %.5f %%)" % (bias, bias * ghi_ratio))
            rmse = error.rmse_es(estimated, measured, s)
            show("RMSE:\t%.5f\t( %.5f %%)" % (rmse, rmse * ghi_ratio))
            mae = error.mae(estimated, measured, s)
            show("MAE:\t%.5f\t( %.5f %%)" % (mae, mae * ghi_ratio))
            show("----------\n")
            error.rmse(root, s)

    def run_with(self, loader):
        self.process_globalradiation(loader, self.cache)
        #    process_validate(root)
        # draw.getpng(draw.matrixtogrey(data[15]),'prueba.png')


class TemporalCache(object):

    def __init__(self, strategy):
        self.strategy = strategy
        self.filenames = self.strategy.filenames
        self.initialize_path(self.filenames)
        self.update_cache(self.filenames)
        self.cache = Loader(map(self.get_cached_file, self.filenames))
        self.root = self.cache.root
        self._attrs = {}

    def initialize_path(self, filenames):
        self.path = '/'.join(filenames[0].split('/')[0:-1])
        self.temporal_path = 'temporal_cache'
        self.index = {self.get_cached_file(v): v for v in filenames}
        if not os.path.exists(self.temporal_path):
            os.makedirs(self.temporal_path)

    def get_cached_file(self, filename):
        return '%s/%s' % (self.temporal_path, short(filename, None, None))

    def update_cache(self, filenames):
        self.clean_cache(filenames)
        self.extend_cache(filenames)

    def extend_cache(self, filenames):
        cached_files = glob.glob('%s/*.nc' % self.temporal_path)
        not_cached = filter(lambda f: self.get_cached_file(f) not in cached_files,
                            filenames)
        if not_cached:
            loader = Loader(not_cached)
            new_files = map(self.get_cached_file, not_cached)
            strategy = self.strategy
            with nc.loader(new_files) as cache:
                strategy.process_temporalcache(loader, cache)

    def clean_cache(self, exceptions):
        cached_files = glob.glob('%s/*.nc' % self.temporal_path)
        old_cache = filter(lambda f: self.index[f] not in exceptions,
                           cached_files)

    def getvar(self, *args, **kwargs):
        tmp = list(args)
        tmp.insert(0, self.cache.root)
        var = nc.getvar(*tmp, **kwargs)
        return var

    def __getattr__(self, name):
        if name not in self._attrs.keys():
            self._attrs[name] = nc.getvar(self.root, name)
        return self._attrs[name]


def filter_filenames(filename):
    files = glob.glob(filename) if isinstance(filename, str) else filename
    last_dt = to_datetime(max(files))
    a_month_ago = (last_dt - timedelta(days=30)).date()
    include_lastmonth = lambda dt: dt.date() > a_month_ago
    files = filter(lambda f: include_lastmonth(to_datetime(f)), files)
    include_daylight = lambda dt: dt.hour - 4 >= 6 and dt.hour - 4 <= 18
    files = filter(lambda f: include_daylight(to_datetime(f)), files)
    return files


def workwith(filename="data/goes13.*.BAND_01.nc"):
    filenames = filter_filenames(filename)
    months = list(set(map(lambda dt: '%i/%i' % (dt.month, dt.year),
                          map(to_datetime, filenames))))
    show("=======================")
    show("Months: ", months)
    show("Dataset: ", len(filenames), " files.")
    show("-----------------------\n")
    loader = Loader(filenames)
    strategy = Heliosat2(filenames)
    strategy.run_with(loader)
    show("Process finished.\n")
