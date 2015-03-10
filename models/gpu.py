import numpy as np
import models.core as gpu
from netcdf import netcdf as nc
import cpu
# gpu.cuda_can_help = False


getslots = cpu.getslots

mod_getexcentricity = gpu.SourceModule(
    """
    __global__ void getexcentricity(float *gamma)
    {
        int it = threadIdx.x;
        int i2d = blockDim.x*blockIdx.x;
        int i3d = i2d + it;
        gamma[i3d] *= """ + gpu.deg2rad_ratio + """;
        gamma[i3d] = 1.000110 + 0.034221 * cos(gamma[i3d]) + \
            0.001280 * sin(gamma[i3d]) + \
            0.000719 * cos(2 * gamma[i3d]) + \
            0.000077 * sin(2 * gamma[i3d]);
    }
    """)


def getexcentricity(gamma):
    func = mod_getexcentricity.get_function("getexcentricity")
    result = gpu.gpu_exec(func, gamma)
    return result


mod_getdeclination = gpu.SourceModule(
    """
    __global__ void getdeclination(float *gamma)
    {
        int it = threadIdx.x;
        int i2d = blockDim.x * blockIdx.x;
        int i3d = i2d + it;
        gamma[i3d] *= """ + gpu.deg2rad_ratio + """;
        gamma[i3d] = 0.006918 - 0.399912 * cos(gamma[i3d]) + \
            0.070257 * sin(gamma[i3d]) - \
            0.006758 * cos(2 * gamma[i3d]) + \
            0.000907 * sin(2 * gamma[i3d]) - \
            0.002697 * cos(3 * gamma[i3d]) + \
            0.00148 * sin(3 * gamma[i3d]);
        gamma[i3d] *= """ + gpu.rad2deg_ratio + """;
    }
    """)


def getdeclination(gamma):
    func = mod_getdeclination.get_function("getdeclination")
    result = gpu.gpu_exec(func, gamma)
    return result


mod_getzenithangle = gpu.SourceModule(
    """
    __global__ void getzenithangle(float *hourlyangle, float *lat, float *dec)
    {
        int it = threadIdx.x;
        int i2d = blockDim.x * blockIdx.x;
        int i3d = i2d + it;
        float lat_r = lat[i2d] * """ + gpu.deg2rad_ratio + """;
        float dec_r = dec[it] * """ + gpu.deg2rad_ratio + """;
        hourlyangle[i3d] *= """ + gpu.deg2rad_ratio + """;
        hourlyangle[i3d] = acos(sin(dec_r) * sin(lat_r) + \
            cos(dec_r) * cos(lat_r) * cos(hourlyangle[i3d]));
        hourlyangle[i3d] *= """ + gpu.rad2deg_ratio + """;
    }
    """)


def getzenithangle(declination, latitude, hourlyangle):
    func = mod_getzenithangle.get_function("getzenithangle")
    result = gpu.gpu_exec(func, hourlyangle, latitude, declination)
    return result


mod_getalbedo = gpu.SourceModule(
    """
    __global__ void getalbedo(float *radiance, float totalirradiance, \
    float *excentricity, float *zenitangle)
    {
        int it = threadIdx.x;
        int i2d = blockDim.x*blockIdx.x;
        int i3d = i2d + it;
        radiance[i3d] = (""" + str(np.float32(np.pi)) + """ * \
            radiance[i3d]) / (totalirradiance * excentricity[i3d] * \
            cos(zenitangle[i3d]));
    }
    """)


def getalbedo(radiance, totalirradiance, excentricity, zenitangle):
    func = mod_getalbedo.get_function("getalbedo")
    result = gpu.gpu_exec(func, radiance, totalirradiance,
                          excentricity, zenitangle)
    return result


mod_getsatellitalzenithangle = gpu.SourceModule(
    """
    __global__ void getsatellitalzenithangle(float *lat, float *lon, \
    float sub_lon, float rpol, float req, float h)
    {
        //int it = threadIdx.x;
        int i2d = blockDim.x*blockIdx.x;
        //int i3d = i2d + it;
        lat[i2d] *= """ + gpu.deg2rad_ratio + """;
        float lon_diff = (lon[i2d] - sub_lon) * """ + gpu.deg2rad_ratio + """;
        float lat_cos_only = cos(lat[i2d]);
        float re = rpol / (sqrt(1 - (pow(req, 2) - pow(rpol, 2)) / \
            (pow(req, 2)) * pow(lat_cos_only, 2)));
        float lat_cos = re * lat_cos_only;
        float r1 = h - lat_cos * cos(lon_diff);
        float r2 = - lat_cos * sin(lon_diff);
        float r3 = re * sin(lat[i2d]);
        float rs = sqrt(pow(r1,2) + pow(r2,2) + pow(r3,2));
        lat[i2d] = (""" + str(np.float32(np.pi)) + """ - acos((pow(h,2) - \
            pow(re, 2) - pow(rs, 2)) / (-2 * re * rs)));
        lat[i2d] *= """ + gpu.rad2deg_ratio + """;
    }
    """)


def getsatellitalzenithangle(lat, lon, sub_lon):
    rpol = 6356.5838
    req = 6378.1690
    h = 42166.55637  # 42164.0
    func = mod_getsatellitalzenithangle.get_function(
        "getsatellitalzenithangle")
    result = gpu.gpu_exec(func, lat, lon, sub_lon, rpol, req, h)
    return result


def obtain_gamma_params(time, lat, lon):
    gamma = cpu.getdailyangle(cpu.getjulianday(time),
                              cpu.gettotaldays(time))
    tst_hour = cpu.gettsthour(cpu.getdecimalhour(time),
                              cpu.GREENWICH_LON, lon,
                              cpu.gettimeequation(gamma))
    declination = getdeclination(gamma)
    omega = cpu.gethourlyangle(tst_hour, lat / abs(lat))
    solarangle = getzenithangle(declination, lat, omega)
    solarelevation = cpu.getelevation(solarangle)
    excentricity = getexcentricity(gamma)
    return declination, solarangle, solarelevation, excentricity


getbeamirradiance = cpu.getbeamirradiance
getdiffuseirradiance = cpu.getdiffuseirradiance
getglobalirradiance = cpu.getglobalirradiance
getatmosphericradiance = cpu.getatmosphericradiance
getelevation = cpu.getelevation
getopticalpath = cpu.getopticalpath
getcorrectedelevation = cpu.getcorrectedelevation
getopticaldepth = cpu.getopticaldepth
gettransmitance = cpu.gettransmitance
geteffectivealbedo = cpu.geteffectivealbedo
getcloudalbedo = cpu.getcloudalbedo
getapparentalbedo = cpu.getapparentalbedo
getsecondmin = cpu.getsecondmin
getsolarelevation = cpu.getsolarelevation
getcloudindex = cpu.getcloudindex
getclearsky = cpu.getclearsky


def process_temporalcache(strategy, loader, cache):
    time = loader.time
    shape = list(time.shape)
    shape.append(1)
    time = time.reshape(tuple(shape))
    nc.getdim(cache, 'xc_k', 1)
    nc.getdim(cache, 'yc_k', 1)
    nc.getdim(cache, 'time', 1)
    slots = cache.getvar('slots', 'i1', ('time', 'yc_k', 'xc_k'))
    slots[:] = getslots(time, strategy.IMAGE_PER_HOUR)
    nc.sync(cache)
    declination = cache.getvar('declination', source=slots)
    solarangle = cache.getvar('solarangle', 'f4', source=loader.ref_data)
    solarelevation = cache.getvar('solarelevation', source=solarangle)
    excentricity = cache.getvar('excentricity', source=slots)
    show("Calculaing time related paramenters... ")
    result = obtain_gamma_params(time, loader.lat[0], loader.lon[0])
    declination[:], solarangle[:], solarelevation[:], excentricity[:] = result
    show("Calculating irradiances... ")
    # The average extraterrestrial irradiance is 1367.0 Watts/meter^2
    bc = getbeamirradiance(1367.0, excentricity[:], solarangle[:],
                               solarelevation[:], loader.linke, loader.dem)
    dc = getdiffuseirradiance(1367.0, excentricity[:], solarelevation[:],
                                  loader.linke)
    gc = getglobalirradiance(bc, dc)
    v_gc = nc.getvar(cache, 'gc', source=loader.ref_data)
    v_gc[:] = gc
    v_gc, v_dc = None, None
    show("Calculating the satellital parameters... ")
    satellitalzenithangle = getsatellitalzenithangle(loader.lat, loader.lon,
                                                     strategy.SAT_LON)
    atmosphericradiance = getatmosphericradiance(1367.0,
                                                 strategy.i0met,
                                                 dc,
                                                 satellitalzenithangle)
    atmosphericalbedo = getalbedo(atmosphericradiance, strategy.i0met,
                                  excentricity[:],
                                  satellitalzenithangle)
    satellitalelevation = getelevation(satellitalzenithangle)
    v_atmosphericalbedo = nc.getvar(cache, 'atmosphericalbedo',
                                    source=loader.ref_data)
    v_atmosphericalbedo[:] = atmosphericalbedo
    show("Calculating satellital optical path and optical depth... ")
    satellital_opticalpath = getopticalpath(
        getcorrectedelevation(satellitalelevation), loader.dem, 8434.5)
    satellital_opticaldepth = getopticaldepth(satellital_opticalpath)
    show("Calculating earth-satellite transmitances... ")
    t_sat = gettransmitance(loader.linke, satellital_opticalpath,
                            satellital_opticaldepth, satellitalelevation)
    v_sat = nc.getvar(cache, 't_sat', source=loader.ref_lon)
    v_sat[:] = t_sat
    v_atmosphericalbedo, v_sat = None, None
    show("Calculating solar optical path and optical depth... ")
    # The maximum height of the non-transparent atmosphere is at 8434.5 mts
    solar_opticalpath = getopticalpath(
        getcorrectedelevation(solarelevation[:]), loader.dem, 8434.5)
    solar_opticaldepth = getopticaldepth(solar_opticalpath)
    show("Calculating sun-earth transmitances... ")
    t_earth = gettransmitance(loader.linke, solar_opticalpath,
                              solar_opticaldepth, solarelevation[:])
    v_earth = nc.getvar(cache, 't_earth', source=loader.ref_data)
    v_earth[:] = t_earth
    v_earth, v_sat = None, None
    effectivealbedo = geteffectivealbedo(solarangle[:])
    cloudalbedo = getcloudalbedo(effectivealbedo, atmosphericalbedo,
                                 t_earth, t_sat)
    v_albedo = nc.getvar(cache, 'cloudalbedo', source=loader.ref_data)
    v_albedo[:] = cloudalbedo
    nc.sync(cache)
    v_albedo = None


def process_globalradiation(strategy, loader, cache):
    excentricity = cache.excentricity[:]
    solarangle = cache.solarangle[:]
    atmosphericalbedo = cache.atmosphericalbedo[:]
    t_earth = cache.t_earth[:]
    t_sat = cache.t_sat[:]
    show("Calculating observed albedo, apparent albedo, effective albedo and "
         "cloud albedo... ")
    observedalbedo = getalbedo(loader.calibrated_data, strategy.i0met,
                               excentricity, solarangle)
    apparentalbedo = getapparentalbedo(observedalbedo, atmosphericalbedo,
                                       t_earth, t_sat)
    declination = cache.declination[:]
    show("Calculating the noon window... ")
    slot_window_in_hours = 4
    # On meteosat are 96 image per day
    image_per_day = 24 * strategy.IMAGE_PER_HOUR
    # and 48 image to the noon
    noon_slot = image_per_day / 2
    half_window = strategy.IMAGE_PER_HOUR * slot_window_in_hours/2
    min_slot = noon_slot - half_window
    max_slot = noon_slot + half_window
    # Create the condition used to work only with the data inside that window
    show("Filtering the data outside the calculated window... ")
    condition = ((cache.slots >= min_slot) & (cache.slots < max_slot))
    # TODO: Meteosat: From 40 to 56 inclusive (the last one is not included)
    condition = np.reshape(condition, condition.shape[0])
    mask1 = loader.calibrated_data[condition] <= (strategy.i0met / np.pi) * 0.03
    m_apparentalbedo = np.ma.masked_array(apparentalbedo[condition], mask1)
    # To do the nexts steps needs a lot of memory
    show("Calculating the ground reference albedo... ")
    mask2 = m_apparentalbedo < stats.scoreatpercentile(m_apparentalbedo, 5)
    p5_apparentalbedo = np.ma.masked_array(m_apparentalbedo, mask2)
    groundreferencealbedo = getsecondmin(p5_apparentalbedo)
    # Calculate the solar elevation using times, latitudes and omega
    show("Calculating solar elevation... ")
    r_alphanoon = getsolarelevation(declination, loader.lat[0], 0)
    r_alphanoon = r_alphanoon * 2./3.
    r_alphanoon[r_alphanoon > 40] = 40
    r_alphanoon[r_alphanoon < 15] = 15
    solarelevation = cache.solarelevation[:]
    show("Calculating the apparent albedo second minimum... ")
    groundminimumalbedo = getsecondmin(
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
    cloudindex = getcloudindex(apparentalbedo, groundminimumalbedo,
                                   cloudalbedo[:])
    apparentalbedo = None
    show("Calculating the clear sky... ")
    clearsky = getclearsky(cloudindex)
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
