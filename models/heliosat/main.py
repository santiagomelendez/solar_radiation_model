#!/usr/bin/env python

import sys
sys.path.append(".")
import numpy as np
from datetime import datetime
from libs.statistics import stats
import glob

from netcdf import netcdf as nc
from libs.geometry import jaen as geo
from linketurbidity import instrument as linke
from noaadem import instrument as dem
# import processgroundstations as pgs
from libs.console import say, show

SAT_LON = -75.113  # -75.3305 # longitude of sub-satellite point in degrees
GREENWICH_LON = 0.0
IMAGE_PER_HOUR = 2


def geti0met():
    GOES_OBSERVED_ALBEDO_CALIBRATION = 1.89544 * (10 ** (-3))
    return np.pi / GOES_OBSERVED_ALBEDO_CALIBRATION


def process_temporal_data(loader):
    lat = loader.lat[0]
    lon = loader.lon[0]
    times = [datetime.utcfromtimestamp(int(t))
             for t in loader.time]
    indexes = range(len(times))
    data = loader.data
    nc.getdim(loader.root, 'xc_k', 1)
    nc.getdim(loader.root, 'yc_k', 1)
    gamma = nc.getvar(loader.root, 'gamma', 'f4',
                      dimensions=('time', 'yc_k', 'xc_k'))
    nc.sync(loader.root)
    tst_hour = nc.getvar(loader.root, 'tst_hour', 'f4', source=data)
    declination = nc.getvar(loader.root, 'declination', source=gamma)
    solarangle = nc.getvar(loader.root, 'solarangle', 'f4', source=data)
    solarelevation = nc.getvar(loader.root, 'solarelevation', source=solarangle)
    excentricity = nc.getvar(loader.root, 'excentricity', source=gamma)
    slots = nc.getvar(loader.root, 'slots', 'i1', ('time', 'yc_k', 'xc_k'))
    nc.sync(loader.root)
    for i in indexes:
        show("\rTemporal data: preprocessing image %d / %d " %
             (i, len(indexes) - 1))
        dt = times[i]
        # Calculate some geometry parameters
        # Parameters that need the datetime:
        # gamma, tst_hour, slots, linketurbidity
        gamma[i] = geo.getdailyangle(geo.getjulianday(dt),
                                     geo.gettotaldays(dt))
        tst_hour[i, :] = geo.gettsthour(
            geo.getdecimalhour(dt),
            GREENWICH_LON,
            lon,
            geo.gettimeequation(gamma[i]))
        declination[i] = geo.getdeclination(gamma[i])
        slots[i] = geo.getslots(dt, IMAGE_PER_HOUR)
        omega = geo.gethourlyangle(tst_hour[i, :], lat/abs(lat))
        solarangle[i, :] = geo.getzenithangle(declination[i], lat, omega)
        solarelevation[i, :] = geo.getelevation(solarangle[i, :])
        excentricity[i] = geo.getexcentricity(gamma[i])
        nc.sync(loader.root)
    say("Calculating the satellital zenith angle... ")
    satellitalzenithangle = geo.getsatellitalzenithangle(lat, lon, SAT_LON)
    v_satellitalzenithangle = nc.getvar(loader.root, 'satellitalzenithangle',
                                        source=loader.lat)
    v_satellitalzenithangle[:] = satellitalzenithangle
    nc.sync(loader.root)
    v_satellitalzenithangle = None


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
    satellitalzenithangle = loader.satellitalzenithangle[:]
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
    v_atmosphericalbedo = nc.getvar(loader.root, 'atmosphericalbedo', source=data)
    v_atmosphericalbedo[:] = atmosphericalbedo
    v_satellitalelevation = nc.getvar(loader.root, 'satellitalelevation', source=lon)
    v_satellitalelevation[:] = satellitalelevation
    nc.sync(loader.root)
    v_atmosphericalbedo, v_satellitalelevation = None, None


def process_optical_fading(loader):
    solarelevation = loader.solarelevation[:]
    terrain = loader.dem[:]
    satellitalelevation = loader.satellitalelevation[:]
    linketurbidity = loader.linke[:]
    say("Calculating optical path and optical depth... ")
    # The maximum height of the non-transparent atmosphere is at 8434.5 mts
    solar_opticalpath = geo.getopticalpath(
        geo.getcorrectedelevation(solarelevation), terrain, 8434.5)
    solar_opticaldepth = geo.getopticaldepth(solar_opticalpath)
    satellital_opticalpath = geo.getopticalpath(
        geo.getcorrectedelevation(satellitalelevation), terrain, 8434.5)
    satellital_opticaldepth = geo.getopticaldepth(satellital_opticalpath)
    say("Calculating sun-earth and earth-satellite transmitances... ")
    t_earth = geo.gettransmitance(linketurbidity, solar_opticalpath,
                                  solar_opticaldepth, solarelevation)
    t_sat = geo.gettransmitance(linketurbidity, satellital_opticalpath,
                                satellital_opticaldepth, satellitalelevation)
    data = loader.data
    satellitalelevation = loader.satellitalelevation
    v_earth = nc.getvar(loader.root, 't_earth', source=data)
    v_earth[:] = t_earth
    v_sat = nc.getvar(loader.root, 't_sat', source=satellitalelevation)
    v_sat[:] = t_sat
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
    v_albedo = nc.getvar(loader.root, 'effectivealbedo', source=data_ref)
    v_albedo[:] = effectivealbedo
    nc.sync(loader.root)
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
    f_var = nc.getvar(loader.root, 'cloudinessindex', source=cloudalbedo)
    f_var[:] = cloudindex
    nc.sync(loader.root)
    say("Calculating the clear sky... ")
    clearsky = geo.getclearsky(cloudindex)
    cloudindex = None
    f_var = nc.getvar(loader.root, 'clearskyindex', source=cloudalbedo)
    f_var[:] = clearsky
    nc.sync(loader.root)
    say("Calculating the global radiation... ")
    clearskyglobalradiation = nc.getvar(loader.root, 'gc')
    globalradiation = clearsky * clearskyglobalradiation[:]
    f_var = nc.getvar(loader.root, 'globalradiation', source=clearskyglobalradiation)
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
    include_daylight = lambda dt: dt.hour - 4 >= 6 and dt.hour - 4 <= 18
    files = filter(lambda f: include_daylight(to_datetime(f)), files)
    return files


class Static(object):

    def __init__(self, filenames):
        # At first it should have: lat, lon, dem, linke
        self.root, is_new = nc.open('static.nc')
        if is_new:
            with nc.loader(filenames[0]) as root_ref:
                self.lat = nc.getvar(root_ref, 'lat')
                self.lon = nc.getvar(root_ref, 'lon')
                nc.getvar(root, 'lat', source=lat)
                nc.getvar(root, 'lon', source=lon)
                self.project_dem()
                self.project_linke()
                nc.sync(root)

    def project_dem(self):
        say("Projecting DEM's map... ")
        dem_var = nc.getvar(self.root, 'dem', 'f4', source=self.lon)
        dem_var[:] = dem.obtain(self.lat[0], self.lon[0])

    def project_linke(self):
        say("Projecting Linke's turbidity index... ")
        dts = map(lambda m: datetime(2014, m, 15), range(1,13))
        linkes = map(lambda dt: linke.obtain(dt, compressed=False), dts)
        linkes = map(lambda l: linke.transform_data(l, self.lat[0], self.lon[0]),
                     linkes)
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

    @property
    def dem(self):
        if not hasattr(self, '_cached_dem'):
            self._cached_dem = nc.getvar(self.cached, 'dem')
        return self._cached_dem

    @property
    def linke(self):
        if not hasattr(self, '_cached_linke'):
            if not hasattr(self, '_linke'):
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
        if not name in self._attrs.keys():
            self._attrs[name] = nc.getvar(self.root, name)
        return self._attrs[name]


def workwith(filename="data/goes13.*.BAND_01.nc"):
    filenames = filter_filenames(filename)
    loader = Loader(filenames)
    months = list(set(map(lambda dt: '%i/%i' % (dt.month, dt.year),
                          map(to_datetime, filenames))))
    show("=======================")
    show("Months: ", months)
    show("Dataset: ", len(filenames), " files.")
    show("-----------------------\n")

    process_temporal_data(loader)
    process_atmospheric_data(loader)

    process_ground_albedo(loader)

    process_radiation(loader)

    #    process_validate(root)
    # draw.getpng(draw.matrixtogrey(data[15]),'prueba.png')
    show("Process finished.\n")
