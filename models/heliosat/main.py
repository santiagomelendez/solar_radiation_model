#!/usr/bin/env python

import sys
sys.path.append(".")
import numpy as np
from datetime import datetime, timedelta
from libs.statistics import stats
import glob
import os

from netcdf import netcdf as nc
from libs.geometry import jaen as geo
from linketurbidity import instrument as linke
from noaadem import instrument as dem
# import processgroundstations as pgs
from libs.console import say, show
from collections import defaultdict

SAT_LON = -75.113  # -75.3305 # longitude of sub-satellite point in degrees
GREENWICH_LON = 0.0
IMAGE_PER_HOUR = 2


def geti0met():
    GOES_OBSERVED_ALBEDO_CALIBRATION = 1.89544 * (10 ** (-3))
    return np.pi / GOES_OBSERVED_ALBEDO_CALIBRATION


def process_temporal_data(loader):
    lat = loader.lat[0]
    lon = loader.lon[0]
    data = loader.data
    nc.getdim(loader.root, 'xc_k', 1)
    nc.getdim(loader.root, 'yc_k', 1)
    slots = nc.getvar(loader.root, 'slots', 'i1', ('time', 'yc_k', 'xc_k'))
    nc.sync(loader.root)
    declination = nc.getvar(loader.root, 'declination', source=slots)
    solarangle = nc.getvar(loader.root, 'solarangle', 'f4', source=data)
    solarelevation = nc.getvar(loader.root, 'solarelevation',
                               source=solarangle)
    excentricity = nc.getvar(loader.root, 'excentricity', source=slots)
    nc.sync(loader.root)
    show("Calculaing time related paramenters... ")
    time = loader.time[:]
    shape = list(time.shape)
    shape.append(1)
    time = time.reshape(tuple(shape))
    gamma = geo.getdailyangle(geo.getjulianday(time),
                                 geo.gettotaldays(time))
    tst_hour = geo.gettsthour(geo.getdecimalhour(time),
                                 GREENWICH_LON, lon,
                                 geo.gettimeequation(gamma))
    slots[:] = geo.getslots(time, IMAGE_PER_HOUR)
    nc.sync(loader.root)
    show("Calculating gamma related parameters...")
    declination[:] = geo.getdeclination(gamma)
    omega = geo.gethourlyangle(tst_hour, lat / abs(lat))
    solarangle[:] = geo.getzenithangle(declination[:], lat, omega)
    solarelevation[:] = geo.getelevation(solarangle[:])
    excentricity[:] = geo.getexcentricity(gamma)
    nc.sync(loader.root)


def process_irradiance(loader):
    excentricity = loader.excentricity[:]
    solarangle = loader.solarangle[:]
    solarelevation = loader.solarelevation[:]
    linketurbidity = loader.linke[:]
    terrain = loader.dem[:]
    say("Calculating beam, diffuse and global irradiance... ")
    # The average extraterrestrial irradiance is 1367.0 Watts/meter^2
    bc = geo.getbeamirradiance(1367.0, excentricity, solarangle,
                               solarelevation, linketurbidity, terrain)
    dc = geo.getdiffuseirradiance(1367.0, excentricity, solarelevation,
                                  linketurbidity)
    gc = geo.getglobalirradiance(bc, dc)
    data = nc.getvar(loader.root, 'data')
    v_bc = nc.getvar(loader.root, 'bc', source=data)
    v_bc[:] = bc
    v_dc = nc.getvar(loader.root, 'dc', source=data)
    v_dc[:] = dc
    v_gc = nc.getvar(loader.root, 'gc', source=data)
    v_gc[:] = gc
    nc.sync(loader.root)
    v_gc, v_dc, v_bc = None, None, None


def process_atmospheric_irradiance(loader):
    i0met = geti0met()
    dc = loader.dc[:]
    say("Calculating the satellital parameters... ")
    lat, lon = loader.lat[:], loader.lon[:]
    satellitalzenithangle = geo.getsatellitalzenithangle(lat, lon, SAT_LON)
    excentricity = loader.excentricity[:]
    say("Calculating atmospheric irradiance... ")
    atmosphericradiance = geo.getatmosphericradiance(1367.0,
                                                     i0met,
                                                     dc,
                                                     satellitalzenithangle)
    atmosphericalbedo = geo.getalbedo(atmosphericradiance, i0met, excentricity,
                                      satellitalzenithangle)
    satellitalelevation = geo.getelevation(satellitalzenithangle)
    data = loader.data
    lon = loader.lon
    v_atmosphericalbedo = nc.getvar(loader.root, 'atmosphericalbedo',
                                    source=data)
    v_atmosphericalbedo[:] = atmosphericalbedo
    say("Calculating satellital optical path and optical depth... ")
    satellital_opticalpath = geo.getopticalpath(
        geo.getcorrectedelevation(satellitalelevation), loader.dem[:], 8434.5)
    satellital_opticaldepth = geo.getopticaldepth(satellital_opticalpath)
    say("Calculating earth-satellite transmitances... ")
    t_sat = geo.gettransmitance(loader.linke[:], satellital_opticalpath,
                                satellital_opticaldepth, satellitalelevation)
    v_sat = nc.getvar(loader.root, 't_sat', source=lon)
    v_sat[:] = t_sat
    nc.sync(loader.root)
    v_atmosphericalbedo, v_sat = None, None


def process_optical_fading(loader):
    solarelevation = loader.solarelevation[:]
    say("Calculating solar optical path and optical depth... ")
    # The maximum height of the non-transparent atmosphere is at 8434.5 mts
    solar_opticalpath = geo.getopticalpath(
        geo.getcorrectedelevation(solarelevation), loader.dem[:], 8434.5)
    solar_opticaldepth = geo.getopticaldepth(solar_opticalpath)
    say("Calculating sun-earth transmitances... ")
    t_earth = geo.gettransmitance(loader.linke[:], solar_opticalpath,
                                  solar_opticaldepth, solarelevation)
    data = loader.data
    v_earth = nc.getvar(loader.root, 't_earth', source=data)
    v_earth[:] = t_earth
    nc.sync(loader.root)
    v_earth, v_sat = None, None


def process_albedos(loader):
    i0met = geti0met()
    excentricity = loader.excentricity[:]
    solarangle = loader.solarangle[:]
    atmosphericalbedo = loader.atmosphericalbedo[:]
    t_earth = loader.t_earth[:]
    t_sat = loader.t_sat[:]
    say("Calculating observed albedo, apparent albedo, effective albedo and "
        "cloud albedo... ")
    observedalbedo = geo.getalbedo(loader.calibrated_data, i0met,
                                   excentricity, solarangle)
    data_ref = loader.data
    v_albedo = nc.getvar(loader.root, 'observedalbedo', source=data_ref)
    v_albedo[:] = observedalbedo
    nc.sync(loader.root)
    apparentalbedo = geo.getapparentalbedo(observedalbedo, atmosphericalbedo,
                                           t_earth, t_sat)
    v_albedo = nc.getvar(loader.root, 'apparentalbedo', source=data_ref)
    v_albedo[:] = apparentalbedo
    nc.sync(loader.root)
    effectivealbedo = geo.geteffectivealbedo(solarangle)
    cloudalbedo = geo.getcloudalbedo(effectivealbedo, atmosphericalbedo,
                                     t_earth, t_sat)
    v_albedo = nc.getvar(loader.root, 'cloudalbedo', source=data_ref)
    v_albedo[:] = cloudalbedo
    nc.sync(loader.root)
    v_albedo = None


def process_atmospheric_data(loader):
    process_irradiance(loader)
    process_atmospheric_irradiance(loader)
    process_optical_fading(loader)
    process_albedos(loader)


def process_ground_albedo(loader):
    slots = loader.slots[:]
    declination = loader.declination[:]
    # The day is divided into _slots_ to avoid the minutes diferences
    # between days.
    # TODO: Related with the solar hour at the noon if the pictures are taken
    # every 15 minutes (meteosat)
    say("Calculating the noon window... ")
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
    say("Filtering the data outside the calculated window... ")
    condition = ((slots >= min_slot) & (slots < max_slot))
    # TODO: Meteosat: From 40 to 56 inclusive (the last one is not included)
    condition = np.reshape(condition, condition.shape[0])
    apparentalbedo = loader.apparentalbedo[:]
    mask1 = loader.calibrated_data[condition] <= (geti0met() / np.pi) * 0.03
    m_apparentalbedo = np.ma.masked_array(apparentalbedo[condition], mask1)
    # To do the nexts steps needs a lot of memory
    say("Calculating the ground reference albedo... ")
    mask2 = m_apparentalbedo < stats.scoreatpercentile(m_apparentalbedo, 5)
    p5_apparentalbedo = np.ma.masked_array(m_apparentalbedo, mask2)
    groundreferencealbedo = geo.getsecondmin(p5_apparentalbedo)
    # Calculate the solar elevation using times, latitudes and omega
    say("Calculating solar elevation... ")
    r_alphanoon = geo.getsolarelevation(declination, loader.lat[0], 0)
    r_alphanoon = r_alphanoon * 2./3.
    r_alphanoon[r_alphanoon > 40] = 40
    r_alphanoon[r_alphanoon < 15] = 15
    solarelevation = loader.solarelevation[:]
    say("Calculating the apparent albedo second minimum... ")
    groundminimumalbedo = geo.getsecondmin(
        np.ma.masked_array(apparentalbedo[condition],
                           solarelevation[condition] < r_alphanoon[condition]))
    aux_2g0 = 2 * groundreferencealbedo
    aux_05g0 = 0.5 * groundreferencealbedo
    condition_2g0 = groundminimumalbedo > aux_2g0
    condition_05g0 = groundminimumalbedo < aux_05g0
    groundminimumalbedo[condition_2g0] = aux_2g0[condition_2g0]
    groundminimumalbedo[condition_05g0] = aux_05g0[condition_05g0]
    say("Synchronizing with the NetCDF4 file... ")
    lat_ref = loader.lat
    f_groundalbedo = nc.getvar(loader.root, 'groundalbedo', source=lat_ref)
    f_groundalbedo[:] = groundminimumalbedo
    nc.sync(loader.root)
    f_groundalbedo = None


def process_radiation(loader):
    apparentalbedo = loader.apparentalbedo[:]
    groundalbedo = loader.groundalbedo[:]
    cloudalbedo = loader.cloudalbedo
    say("Calculating the cloud index... ")
    cloudindex = geo.getcloudindex(apparentalbedo, groundalbedo,
                                   cloudalbedo[:])
    apparentalbedo = None
    groundalbedo = None
    say("Calculating the clear sky... ")
    clearsky = geo.getclearsky(cloudindex)
    # f_var = nc.getvar(loader.root, 'clearskyindex', source=cloudalbedo)
    # f_var[:] = clearsky
    # nc.sync(loader.root)
    say("Calculating the global radiation... ")
    clearskyglobalradiation = nc.getvar(loader.root, 'gc')
    globalradiation = clearsky * clearskyglobalradiation[:]
    f_var = nc.getvar(loader.root, 'globalradiation',
                      source=clearskyglobalradiation)
    say("Saving the global radiation... ")
    f_var[:] = globalradiation
    nc.sync(loader.root)
    f_var = None
    cloudalbedo = None


def process_validate(root):
    from libs.statistics import error
    estimated = nc.getvar(root, 'globalradiation')
    measured = nc.getvar(root, 'measurements')
    stations = [0]
    for s in stations:
        show("==========\n")
        say("Station %i (%i slots)" % (s,  measured[:, s, 0].size))
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


def to_datetime(filename):
    short = (lambda f, start=1, end=-2:
             ".".join((f.split('/')[-1]).split('.')[start:end]))
    return datetime.strptime(short(filename), '%Y.%j.%H%M%S')


def filter_filenames(filename):
    files = glob.glob(filename) if isinstance(filename, str) else filename
    last_dt = to_datetime(max(files))
    a_month_ago = (last_dt - timedelta(days=30)).date()
    include_lastmonth = lambda dt: dt.date() > a_month_ago
    files = filter(lambda f: include_lastmonth(to_datetime(f)), files)
    include_daylight = lambda dt: dt.hour - 4 >= 6 and dt.hour - 4 <= 18
    files = filter(lambda f: include_daylight(to_datetime(f)), files)
    return files


class Static(object):

    def __init__(self, filenames):
        # At first it should have: lat, lon, dem, linke
        self.root, is_new = nc.open('static.nc')
        if is_new:
            say("This is the first execution from the deployment... ")
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
        say("Projecting DEM's map... ")
        dem_var = nc.getvar(self.root, 'dem', 'f4', source=self.lon)
        dem_var[:] = dem.obtain(self.lat[0], self.lon[0])

    def project_linke(self):
        say("Projecting Linke's turbidity index... ")
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
        self.static = Static(filenames)
        self.cached = self.static.root
        self._attrs = {}
        self.freq = defaultdict(int)

    @property
    def dem(self):
        if not hasattr(self, '_cached_dem'):
            self._cached_dem = nc.getvar(self.cached, 'dem')
        return self._cached_dem

    @property
    def linke(self):
        if not hasattr(self, '_cached_linke'):
            self._linke = nc.getvar(self.cached, 'linke')
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
            normalized_data = (np.float32(row_data)
                               / counts_shift
                               - space_measurement)
            self._cached_calibrated_data = (normalized_data
                                            * postlaunch
                                            * prelaunch)
        return self._cached_calibrated_data

    def __getattr__(self, name):
        self.freq[name] += 1
        if name not in self._attrs.keys():
            self._attrs[name] = nc.getvar(self.root, name)
        return self._attrs[name]


class VolatileCache(object):

    def __init__(self, filenames):
        self.initialize_path(filenames)
        self.update_cache(filenames)
        self.loader = Loader(filenames)

    def initialize_path(self, filenames):
        self.path = '/'.join(filenames[0].split('/')[0:-1])
        self.volatile_path = 'volatile_cache'
        self.index = {self.get_cached_file(v): v for v in filenames}
        if not os.path.exists(self.volatile_path):
            os.makedirs(self.volatile_path)

    def get_cached_file(self, filename):
        short = (lambda f, start=None, end=None:
                 ".".join((f.split('/')[-1]).split('.')[start:end]))
        return '%s/%s' % (self.volatile_path, short(filename))

    def update_cache(self, filenames):
        cached_files = glob.glob('%s/*.nc' % self.volatile_path)
        not_cached = filter(lambda f: self.get_cached_file(f) not in cached_files,
                            filenames)
        if not_cached:
            loader = Loader(not_cached)
            new_files = map(self.get_cached_file, not_cached)
            with nc.loader(new_files) as cache:
                pass


def workwith(filename="data/goes13.*.BAND_01.nc"):
    filenames = filter_filenames(filename)
    months = list(set(map(lambda dt: '%i/%i' % (dt.month, dt.year),
                          map(to_datetime, filenames))))
    show("=======================")
    show("Months: ", months)
    show("Dataset: ", len(filenames), " files.")
    show("-----------------------\n")
    cache = VolatileCache(filenames)
    loader = cache.loader

    process_temporal_data(loader)
    process_atmospheric_data(loader)

    process_ground_albedo(loader)

    process_radiation(loader)

    #    process_validate(root)
    # draw.getpng(draw.matrixtogrey(data[15]),'prueba.png')
    show("Process finished.\n")
    print loader.freq
