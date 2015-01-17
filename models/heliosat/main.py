#!/usr/bin/env python

import sys
sys.path.append(".")
import numpy as np
from datetime import datetime
from libs.statistics import stats
#import pylab as pl
#from osgeo import gdal

from netcdf import netcdf as nc
import numpy as np
from libs.geometry import jaen as geo
from libs.linke import toolbox as linke
from libs.dem import dem
#from libs.paint import jaen as draw
#import processgroundstations as pgs
from libs.console import say, show, show_times

SAT_LON = -75.113 # -75.3305 # longitude of sub-satellite point in degrees
GREENWICH_LON = 0.0
IMAGE_PER_HOUR = 2

def getsatelliteradiance(data, index):
    return data[index]

def geti0met():
    GOES_OBSERVED_ALBEDO_CALIBRATION = 1.89544 * (10 ** (-3))
    return np.pi / GOES_OBSERVED_ALBEDO_CALIBRATION

def calibrated_data(root):
    data = nc.getvar(root, 'data')
    adapt = lambda m: np.expand_dims(m, axis=len(m.shape)-1)
    counts_shift = adapt(nc.getvar(root, 'counts_shift')[:])
    space_measurement = adapt(nc.getvar(root, 'space_measurement')[:])
    prelaunch = adapt(nc.getvar(root, 'prelaunch_0')[:])
    postlaunch = adapt(nc.getvar(root, 'postlaunch')[:])
    print data.shape, counts_shift.shape, space_measurement.shape, prelaunch.shape, postlaunch.shape
    # INFO: Without the postlaunch coefficient the RMSE go to 15%
    return postlaunch * prelaunch * (np.float32(data[:]) /
                                     counts_shift - space_measurement)

def process_temporal_data(lat, lon, root):
    times = [ datetime.utcfromtimestamp(int(t)) for t in nc.getvar(root, 'time') ]
    indexes = range(len(times))
    data = nc.getvar(root, 'data')
    gamma = nc.getvar(root, 'gamma', 'f4', dimensions=('time', ))
    nc.sync(root)
    tst_hour = nc.getvar(root, 'tst_hour', 'f4', source=data)
    declination = nc.getvar(root, 'declination', source=gamma)
    solarangle = nc.getvar(root, 'solarangle', 'f4', source=data)
    solarelevation = nc.getvar(root, 'solarelevation', source=solarangle)
    excentricity = nc.getvar(root, 'excentricity', source=gamma)
    slots = nc.getvar(root, 'slots', 'i1', ('time',))
    nc.sync(root)
    for i in indexes:
        show("\rTemporal data: preprocessing image %d / %d " % (i, len(indexes)-1))
        dt = times[i]
        # Calculate some geometry parameters
        # Parameters that need the datetime: gamma, tst_hour, slots, linketurbidity
        gamma[i] = geo.getdailyangle(geo.getjulianday(dt),geo.gettotaldays(dt))
        tst_hour[i,:] = geo.gettsthour(
            geo.getdecimalhour(dt),
            GREENWICH_LON,
            lon,
            geo.gettimeequation(gamma[i]))
        declination[i] = geo.getdeclination(gamma[i])
        slots[i] = geo.getslots(dt,IMAGE_PER_HOUR)
        omega = geo.gethourlyangle(tst_hour[i,:], lat/abs(lat))
        solarangle[i,:] = geo.getzenithangle(declination[i],lat,omega)
        solarelevation[i,:] = geo.getelevation(solarangle[i,:])
        excentricity[i] = geo.getexcentricity(gamma[i])
        nc.sync(root)

    #nc.sync(root)
    say("Projecting Linke's turbidity index... ")
    linke.cut_projected(root)
    say("Calculating the satellital zenith angle... ")
    satellitalzenithangle = geo.getsatellitalzenithangle(lat, lon, SAT_LON)
    dem.cut_projected(root)
    v_satellitalzenithangle = nc.clonevar(root,'lat', 'satellitalzenithangle')
    v_satellitalzenithangle[:] = satellitalzenithangle
    nc.sync(root)
    v_satellitalzenithangle = None

def process_irradiance(root):
    excentricity = nc.getvar(root,'excentricity')[:]
    solarangle = nc.getvar(root,'solarangle')[:]
    solarelevation = nc.getvar(root,'solarelevation')[:]
    linketurbidity = nc.getvar(root,'linketurbidity')[0]
    terrain = nc.getvar(root,'dem')[:]
    say("Calculating beam, diffuse and global irradiance... ")
    # The average extraterrestrial irradiance is 1367.0 Watts/meter^2
    bc = geo.getbeamirradiance(1367.0,excentricity,solarangle,solarelevation,linketurbidity,terrain)
    dc = geo.getdiffuseirradiance(1367.0,excentricity,solarelevation,linketurbidity)
    gc = geo.getglobalirradiance(bc,dc)
    v_bc = nc.clonevar(root, 'data', 'bc')
    v_bc[:] = bc
    v_dc = nc.clonevar(root, 'data', 'dc')
    v_dc[:] = dc
    v_gc = nc.clonevar(root, 'data', 'gc')
    v_gc[:] = gc
    nc.sync(root)
    v_gc, v_dc, v_bc = None, None, None

def process_atmospheric_irradiance(root):
    i0met = geti0met()
    dc = nc.getvar(root,'dc')[:]
    satellitalzenithangle = nc.getvar(root,'satellitalzenithangle')[:]
    excentricity = nc.getvar(root,'excentricity')[:]
    say("Calculating atmospheric irradiance... ")
    atmosphericradiance = geo.getatmosphericradiance(1367.0, i0met,dc, satellitalzenithangle)
    atmosphericalbedo = geo.getalbedo(atmosphericradiance, i0met, excentricity, satellitalzenithangle)
    satellitalelevation = geo.getelevation(satellitalzenithangle)
    v_atmosphericalbedo = nc.clonevar(root, 'data', 'atmosphericalbedo')
    v_atmosphericalbedo[:] = atmosphericalbedo
    v_satellitalelevation = nc.clonevar(root, 'lon', 'satellitalelevation')
    v_satellitalelevation[:] = satellitalelevation
    nc.sync(root)
    v_atmosphericalbedo, v_satellitalelevation = None, None

def process_optical_fading(root):
    solarelevation = nc.getvar(root,'solarelevation')[:]
    terrain = nc.getvar(root,'dem')[:]
    satellitalelevation = nc.getvar(root, 'satellitalelevation')[:]
    linketurbidity = nc.getvar(root,'linketurbidity')[0]
    say("Calculating optical path and optical depth... ")
    # The maximum height of the non-transparent atmosphere is at 8434.5 mts
    solar_opticalpath = geo.getopticalpath(geo.getcorrectedelevation(solarelevation),terrain, 8434.5)
    solar_opticaldepth = geo.getopticaldepth(solar_opticalpath)
    satellital_opticalpath = geo.getopticalpath(geo.getcorrectedelevation(satellitalelevation),terrain, 8434.5)
    satellital_opticaldepth = geo.getopticaldepth(satellital_opticalpath)
    say("Calculating sun-earth and earth-satellite transmitances... ")
    t_earth = geo.gettransmitance(linketurbidity, solar_opticalpath, solar_opticaldepth, solarelevation)
    t_sat = geo.gettransmitance(linketurbidity, satellital_opticalpath, satellital_opticaldepth, satellitalelevation)
    v_earth = nc.clonevar(root, 'data', 't_earth')
    v_earth[:] = t_earth
    v_sat = nc.clonevar(root, 'satellitalelevation', 't_sat')
    v_sat[:] = t_sat
    nc.sync(root)
    v_earth, v_sat = None, None

def process_albedos(data, root):
    i0met = geti0met()
    excentricity = nc.getvar(root,'excentricity')[:]
    solarangle = nc.getvar(root,'solarangle')[:]
    atmosphericalbedo = nc.getvar(root,'atmosphericalbedo')[:]
    t_earth = nc.getvar(root,'t_earth')[:]
    t_sat = nc.getvar(root,'t_sat')[:]
    say("Calculating observed albedo, apparent albedo, effective albedo and cloud albedo... ")
    observedalbedo = geo.getalbedo(data, i0met , excentricity, solarangle)
    v_albedo = nc.clonevar(root, 'data', 'observedalbedo')
    v_albedo[:] = observedalbedo
    nc.sync(root)
    apparentalbedo = geo.getapparentalbedo(observedalbedo,atmosphericalbedo, t_earth, t_sat)
    v_albedo = nc.clonevar(root, 'data', 'apparentalbedo')
    v_albedo[:] = apparentalbedo
    nc.sync(root)
    effectivealbedo = geo.geteffectivealbedo(solarangle)
    v_albedo = nc.clonevar(root, 'data', 'effectivealbedo')
    v_albedo[:] = effectivealbedo
    nc.sync(root)
    cloudalbedo = geo.getcloudalbedo(effectivealbedo,atmosphericalbedo,t_earth,t_sat)
    v_albedo = nc.clonevar(root, 'data', 'cloudalbedo')
    v_albedo[:] = cloudalbedo
    nc.sync(root)
    v_albedo = None

def process_atmospheric_data(data, root):
    process_irradiance(root)
    process_atmospheric_irradiance(root)
    process_optical_fading(root)
    process_albedos(data, root)

def process_ground_albedo(lat, data, root):
    slots = nc.getvar(root, "slots")[:]
    declination = nc.getvar(root, "declination")[:]
    #The day is divided in _slots_ to avoid the minutes diferences between days.
    # TODO: Related with the solar hour at the noon if the pictures are taken every 15 minutes (meteosat)
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
    condition = ((slots >= min_slot) & (slots < max_slot)) # TODO: Meteosat: From 40 to 56 inclusive (the last one is not included)
    apparentalbedo = nc.getvar(root, "apparentalbedo")[:]
    m_apparentalbedo = np.ma.masked_array(apparentalbedo[condition], data[condition] <= (geti0met()/np.pi) * 0.03)
    # To do the nexts steps needs a lot of memory
    say("Calculating the ground reference albedo... ")
    # TODO: Should review the p5_apparentalbedo parameters and shapes
    p5_apparentalbedo = np.ma.masked_array(m_apparentalbedo, m_apparentalbedo < stats.scoreatpercentile(m_apparentalbedo, 5))
    groundreferencealbedo = geo.getsecondmin(p5_apparentalbedo)
    # Calculate the solar elevation using times, latitudes and omega
    say("Calculating solar elevation... ")
    r_alphanoon = geo.getsolarelevation(declination, lat, 0)
    r_alphanoon = r_alphanoon * 2./3.
    r_alphanoon[r_alphanoon > 40] = 40
    r_alphanoon[r_alphanoon < 15] = 15
    solarelevation = nc.getvar(root, "solarelevation")[:]
    say("Calculating the apparent albedo second minimum... ")
    groundminimumalbedo = geo.getsecondmin(np.ma.masked_array(apparentalbedo[condition], solarelevation[condition] < r_alphanoon[condition]))
    aux_2g0 = 2 * groundreferencealbedo
    aux_05g0 = 0.5 * groundreferencealbedo
    groundminimumalbedo[groundminimumalbedo > aux_2g0] = aux_2g0[groundminimumalbedo > aux_2g0]
    groundminimumalbedo[groundminimumalbedo < aux_05g0] = aux_05g0[groundminimumalbedo < aux_05g0]
    say("Synchronizing with the NetCDF4 file... ")
    f_groundalbedo = nc.clonevar(root, 'lat', 'groundalbedo')
    f_groundalbedo[:] = groundminimumalbedo
    nc.sync(root)
    f_groundalbedo = None

def process_radiation(root):
    apparentalbedo = nc.getvar(root, "apparentalbedo")[:]
    groundalbedo = nc.getvar(root, "groundalbedo")[:]
    cloudalbedo = nc.getvar(root, "cloudalbedo")[:]
    say("Calculating the cloud index... ")
    cloudindex = geo.getcloudindex(apparentalbedo, groundalbedo, cloudalbedo)
    apparentalbedo = None
    groundalbedo = None
    cloudalbedo = None
    f_var = nc.clonevar(root, 'cloudalbedo', 'cloudinessindex')
    f_var[:] = cloudindex
    nc.sync(root)
    say("Calculating the clear sky... ")
    clearsky = geo.getclearsky(cloudindex)
    cloudindex = None
    f_var = nc.clonevar(root, 'cloudalbedo', 'clearskyindex')
    f_var[:] = clearsky
    nc.sync(root)
    say("Calculating the global radiation... ")
    clearskyglobalradiation = nc.getvar(root, 'gc')[:]
    globalradiation = clearsky * clearskyglobalradiation
    f_var = nc.clonevar(root, 'gc', 'globalradiation')
    say("Saving the global radiation... ")
    f_var[:] = globalradiation
    nc.sync(root)
    f_var = None


def process_validate(root):
    from libs.statistics import error
    estimated = nc.getvar(root, 'globalradiation')
    measured = nc.getvar(root, 'measurements')
    stations = [0]
    for s in stations:
        show("==========\n")
        say("Station %i (%i slots)" % (s,  measured[:,s,0].size))
        show("----------")
        show("mean (measured):\t", error.ghi_mean(measured, s))
        show("mean (estimated):\t", estimated[:,s,0].mean())
        ghi_ratio = error.ghi_ratio(measured, s)
        bias = error.bias(estimated, measured, s)
        show("BIAS:\t%.5f\t( %.5f %%)" % (bias, bias * ghi_ratio))
        rmse = error.rmse_es(estimated, measured, s)
        show("RMSE:\t%.5f\t( %.5f %%)" % (rmse, rmse * ghi_ratio))
        mae = error.mae(estimated, measured, s)
        show("MAE:\t%.5f\t( %.5f %%)" % (mae, mae * ghi_ratio))
        show("----------\n")
        error.rmse(root, s)

def workwith(year=2011, month=05, filename="goes13.all.BAND_02.nc"):
    show("=======================")
    show("Year: " , year)
    show("Month: " , month)
    show("Filename: ", filename)
    show("-----------------------\n")

    root = nc.open(filename)[0]
    lat = nc.getvar(root, 'lat')[0]
    lon = nc.getvar(root, 'lon')[0]
    data = calibrated_data(root)

    process_temporal_data(lat, lon, root)
    process_atmospheric_data(data, root)

    process_ground_albedo(lat, data, root)

    process_radiation(root)

    #    process_validate(root)
    #draw.getpng(draw.matrixtogrey(data[15]),'prueba.png')
    nc.close(root)
    show("Process finished.\n")
