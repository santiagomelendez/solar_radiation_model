import numpy as np
from datetime import datetime


def getjulianday(times):
    dts = map(lambda t: datetime.utcfromtimestamp(int(t)), times)
    result = map(lambda dt: dt.timetuple()[7], dts)
    result = np.array(result).reshape(times.shape)
    return result


def gettotaldays(times):
    dts = map(lambda t: datetime.utcfromtimestamp(int(t)), times)
    result = map(lambda dt: datetime(dt.year, 12, 31).timetuple()[7], dts)
    result = np.array(result).reshape(times.shape)
    return result


def getdailyangle(julianday, totaldays):
    return np.rad2deg(2 * np.pi * (julianday - 1) / totaldays)



def getexcentricity(gamma):
    gamma = np.deg2rad(gamma)
    result = (1.000110 + 0.034221 * np.cos(gamma) +
              0.001280 * np.sin(gamma) +
              0.000719 * np.cos(2 * gamma) +
              0.000077 * np.sin(2 * gamma))
    return result


def getdeclination(gamma):
    gamma = np.deg2rad(gamma)
    result = np.rad2deg(0.006918 - 0.399912 * np.cos(gamma) +
                        0.070257 * np.sin(gamma) -
                        0.006758 * np.cos(2 * gamma) +
                        0.000907 * np.sin(2 * gamma) -
                        0.002697 * np.cos(3 * gamma) +
                        0.00148 * np.sin(3 * gamma))
    return result


def gettimeequation(gamma):
    gamma = np.deg2rad(gamma)
    return np.rad2deg((0.000075 + 0.001868 * np.cos(gamma) -
                       0.032077 * np.sin(gamma) -
                       0.014615 * np.cos(2 * gamma) -
                       0.04089 * np.sin(2 * gamma)) * (12 / np.pi))


def getdecimalhour(times):
    dts = map(lambda (idx, t): datetime.utcfromtimestamp(int(t)),
              np.ndenumerate(times))
    result = map(lambda dt: dt.hour + dt.minute/60.0 + dt.second/3600.0, dts)
    return np.array(result).reshape(times.shape)


def gettsthour(hour, d_ref, d, timeequation):
    timeequation = np.deg2rad(timeequation)
    lon_diff = np.deg2rad(d_ref - d)
    return hour - lon_diff * (12 / np.pi) + timeequation


def gethourlyangle(tst_hour, latitud_sign):
    return np.rad2deg((tst_hour - 12) * latitud_sign * np.pi / 12)


def getzenithangle(declination, latitude, hourlyangle):
    hourlyangle = np.deg2rad(hourlyangle)
    lat = np.deg2rad(latitude)
    dec = np.deg2rad(declination)
    result = np.rad2deg(np.arccos(np.sin(dec) * np.sin(lat) +
                                  np.cos(dec) * np.cos(lat) *
                                  np.cos(hourlyangle)))
    return result


def getelevation(zenithangle):
    zenithangle = np.deg2rad(zenithangle)
    return np.rad2deg((np.pi / 2) - zenithangle)


def getsolarelevationmatrix(dt, sub_lon, lat, lon):
    gamma = getdailyangle(getjulianday(dt), gettotaldays(dt))
    declination = getdeclination(gamma)
    timeequation = gettimeequation(gamma)
    tst_hour = gettsthour(getdecimalhour(dt), sub_lon, lon, timeequation)
    omega = gethourlyangle(tst_hour, lat/abs(lat))
    solarangle = getzenithangle(declination, lat, omega)
    return getelevation(solarangle)


def getcorrectedelevation(elevation):
    elevation = np.deg2rad(elevation)
    return np.rad2deg(elevation
                      + 0.061359 * ((0.1594 + 1.1230 * elevation +
                                     0.065656 * np.power(elevation, 2)) /
                                    (1 + 28.9344 * elevation +
                                     277.3971 * np.power(elevation, 2))))


def getopticalpath(correctedelevation, terrainheight,
                   atmosphere_theoretical_height):
    # In the next line the correctedelevation is used over a degree base
    power = np.power(correctedelevation+6.07995, -1.6364)
    # Then should be used over a radian base
    correctedelevation = np.deg2rad(correctedelevation)
    return (np.exp(-terrainheight/atmosphere_theoretical_height) /
            (np.sin(correctedelevation) + 0.50572 * power))


def getopticaldepth(opticalpath):
    tmp = np.zeros(opticalpath.shape) + 1.0
    highslopebeam = opticalpath <= 20
    lowslopebeam = opticalpath > 20
    tmp[highslopebeam] = (6.6296 + 1.7513 * opticalpath[highslopebeam] -
                          0.1202 * np.power(opticalpath[highslopebeam], 2) +
                          0.0065 * np.power(opticalpath[highslopebeam], 3) -
                          0.00013 * np.power(opticalpath[highslopebeam], 4))
    tmp[highslopebeam] = 1 / tmp[highslopebeam]
    tmp[lowslopebeam] = 1 / (10.4 + 0.718 * opticalpath[lowslopebeam])
    return tmp


def getbeamtransmission(linketurbidity, opticalpath, opticaldepth):
    return np.exp(-0.8662 * linketurbidity * opticalpath * opticaldepth)


def gethorizontalirradiance(extraterrestrialirradiance, excentricity,
                            zenitangle):
    zenitangle = np.deg2rad(zenitangle)
    return extraterrestrialirradiance * excentricity * np.cos(zenitangle)


def getbeamirradiance(extraterrestrialirradiance, excentricity, zenitangle,
                      solarelevation, linketurbidity, terrainheight):
    correctedsolarelevation = getcorrectedelevation(solarelevation)
    # TODO: Meteosat is at 8434.5 mts
    opticalpath = getopticalpath(correctedsolarelevation, terrainheight,
                                 8434.5)
    opticaldepth = getopticaldepth(opticalpath)
    return (gethorizontalirradiance(extraterrestrialirradiance,
                                    excentricity, zenitangle)
            * getbeamtransmission(linketurbidity, opticalpath, opticaldepth))


def getzenithdiffusetransmitance(linketurbidity):
    return (-0.015843 + 0.030543 * linketurbidity +
            0.0003797 * np.power(linketurbidity, 2))


def getangularcorrection(solarelevation, linketurbidity):
    solarelevation = np.deg2rad(solarelevation)
    a0 = (0.264631 - 0.061581 * linketurbidity +
          0.0031408 * np.power(linketurbidity, 2))
    a1 = (2.0402 + 0.018945 * linketurbidity -
          0.011161 * np.power(linketurbidity, 2))
    a2 = (-1.3025 + 0.039231 * linketurbidity +
          0.0085079 * np.power(linketurbidity, 2))
    zenitdiffusetransmitance = getzenithdiffusetransmitance(linketurbidity)
    c = a0*zenitdiffusetransmitance < 0.002
    a0[c] = 0.002 / zenitdiffusetransmitance[c]
    return (a0 + a1 * np.sin(solarelevation)
            + a2 * np.power(np.sin(solarelevation), 2))


def getdiffusetransmitance(linketurbidity, solarelevation):
    return (getzenithdiffusetransmitance(linketurbidity) *
            getangularcorrection(solarelevation, linketurbidity))


def gettransmitance(linketurbidity, opticalpath, opticaldepth, solarelevation):
    return (getbeamtransmission(linketurbidity, opticalpath, opticaldepth) +
            getdiffusetransmitance(linketurbidity, solarelevation))


def getdiffuseirradiance(extraterrestrialirradiance, excentricity,
                         solarelevation, linketurbidity):
    return (extraterrestrialirradiance * excentricity *
            getdiffusetransmitance(linketurbidity, solarelevation))


def getglobalirradiance(beamirradiance, diffuseirradiance):
    return beamirradiance + diffuseirradiance


def getalbedo(radiance, totalirradiance, excentricity, zenitangle):
    zenitangle = np.deg2rad(zenitangle)
    result = (np.pi * radiance) / (totalirradiance * excentricity
                                   * np.cos(zenitangle))
    return result


def getsatellitalzenithangle(lat, lon, sub_lon):
    result = None
    rpol = 6356.5838
    req = 6378.1690
    h = 42166.55637  # 42164.0
    lat = np.deg2rad(lat)
    lon_diff = np.deg2rad(lon - sub_lon)
    lat_cos_only = np.cos(lat)
    re = (rpol / (np.sqrt(1 - (req ** 2 - rpol ** 2) / (req ** 2) *
                          np.power(lat_cos_only, 2))))
    lat_cos = re * lat_cos_only
    r1 = h - lat_cos * np.cos(lon_diff)
    r2 = - lat_cos * np.sin(lon_diff)
    r3 = re * np.sin(lat)
    rs = np.sqrt(r1 ** 2 + r2 ** 2 + r3 ** 2)
    result = np.rad2deg(np.pi - np.arccos((h ** 2 - re ** 2 - rs ** 2) /
                                          (-2 * re * rs)))
    return result


def getatmosphericradiance(extraterrestrialirradiance, i0met,
                           diffuseclearsky, satellitalzenitangle):
    satellitalzenitangle = np.deg2rad(satellitalzenitangle)
    anglerelation = np.power((0.5 / np.cos(satellitalzenitangle)), 0.8)
    return ((i0met * diffuseclearsky * anglerelation)
            / (np.pi * extraterrestrialirradiance))


def getdifferentialalbedo(firstalbedo, secondalbedo, t_earth, t_sat):
    return (firstalbedo - secondalbedo) / (t_earth * t_sat)


def getapparentalbedo(observedalbedo, atmosphericalbedo, t_earth, t_sat):
    apparentalbedo = getdifferentialalbedo(observedalbedo, atmosphericalbedo,
                                           t_earth, t_sat)
    apparentalbedo[apparentalbedo < 0] = 0.0
    return apparentalbedo


def geteffectivealbedo(solarangle):
    solarangle = np.deg2rad(solarangle)
    return 0.78 - 0.13 * (1 - np.exp(-4 * np.power(np.cos(solarangle), 5)))


def getcloudalbedo(effectivealbedo, atmosphericalbedo, t_earth, t_sat):
    cloudalbedo = getdifferentialalbedo(effectivealbedo, atmosphericalbedo,
                                        t_earth, t_sat)
    cloudalbedo[cloudalbedo < 0.2] = 0.2
    effectiveproportion = 2.24 * effectivealbedo
    condition = cloudalbedo > effectiveproportion
    cloudalbedo[condition] = effectiveproportion[condition]
    return cloudalbedo


def getslots(times, images_per_hour):
    return np.round(getdecimalhour(times) * images_per_hour).astype(int)


def getintfromdatetime(dt):
    if np.iterable(dt) == 0:
        return int(dt.strftime("%Y%m%d%H%M%S"))
    else:
        return np.array([getintfromdatetime(n) for n in dt])


def getdatetimefromint(number):
    if np.iterable(number) == 0:
        return datetime.strptime(str(number), "%Y%m%d%H%M%S")
    else:
        return np.array([getdatetimefromint(n) for n in number])


def getsolarelevation(declination, lat, omega):
    omega = np.deg2rad(omega)
    declination = np.deg2rad(declination)
    lat = np.deg2rad(lat)
    return np.rad2deg(np.arcsin(np.sin(declination) * np.sin(lat) +
                                np.cos(declination) * np.sin(lat) *
                                np.cos(omega)))


def getsecondmin(albedo):
    min1_albedo = np.ma.masked_array(albedo, albedo == np.amin(albedo, axis=0))
    return np.amin(min1_albedo, axis=0)


def getcloudindex(apparentalbedo, groundalbedo, cloudalbedo):
    return (apparentalbedo - groundalbedo) / (cloudalbedo - groundalbedo)


def getclearsky(cloudindex):
    clearsky = np.zeros_like(cloudindex)
    cond = cloudindex < -0.2
    clearsky[cond] = 1.2
    cond = ((cloudindex >= -0.2) & (cloudindex < 0.8))
    clearsky[cond] = 1 - cloudindex[cond]
    cond = ((cloudindex >= 0.8) & (cloudindex < 1.1))
    clearsky[cond] = (31 - 55 * cloudindex[cond] +
                      25 * np.power(cloudindex[cond], 2)) / 15
    cond = (cloudindex >= 1.1)
    clearsky[cond] = 0.05
    return clearsky


def gettstdatetime(timestamp, tst_hour):
    return np.trunc(timestamp) + tst_hour / 24.
