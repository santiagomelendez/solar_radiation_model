import numpy as np
import stats
from core import ProcessingStrategy
import logging


GREENWICH_LON = 0.0


class CPUStrategy(ProcessingStrategy):

    def getexcentricity(self, gamma):
        gamma = np.deg2rad(gamma)
        return (1.000110 + 0.034221 * np.cos(gamma) +
                0.001280 * np.sin(gamma) +
                0.000719 * np.cos(2 * gamma) +
                0.000077 * np.sin(2 * gamma))

    def getdeclination(self, gamma):
        gamma = np.deg2rad(gamma)
        return np.rad2deg(0.006918 - 0.399912 * np.cos(gamma) +
                          0.070257 * np.sin(gamma) -
                          0.006758 * np.cos(2 * gamma) +
                          0.000907 * np.sin(2 * gamma) -
                          0.002697 * np.cos(3 * gamma) +
                          0.00148 * np.sin(3 * gamma))

    def gettimeequation(self, gamma):
        gamma = np.deg2rad(gamma)
        return np.rad2deg((0.000075 + 0.001868 * np.cos(gamma) -
                           0.032077 * np.sin(gamma) -
                           0.014615 * np.cos(2 * gamma) -
                           0.04089 * np.sin(2 * gamma)) * (12 / np.pi))

    def gettsthour(self, hour, d_ref, d, timeequation):
        timeequation = np.deg2rad(timeequation)
        lon_diff = np.deg2rad(d_ref - d)
        return hour - lon_diff * (12 / np.pi) + timeequation

    def gethourlyangle(self, lat, lon, decimalhour, gamma):
        latitud_sign = lat / abs(lat)
        tst_hour = self.gettsthour(decimalhour, GREENWICH_LON, lon,
                                   self.gettimeequation(gamma))
        return np.rad2deg((tst_hour - 12) * latitud_sign * np.pi / 12)

    def getzenithangle(self, declination, latitude, hourlyangle):
        hourlyangle = np.deg2rad(hourlyangle)
        lat = np.deg2rad(latitude)
        dec = np.deg2rad(declination)
        return np.rad2deg(np.arccos(np.sin(dec) * np.sin(lat) +
                          np.cos(dec) * np.cos(lat) *
                          np.cos(hourlyangle)))

    def getcalibrateddata(self, loader):
        if not hasattr(self, '_cached_calibrated_data'):
            raw_data = loader.data[:]
            counts_shift = loader.counts_shift[:]
            space_measurement = loader.space_measurement[:]
            prelaunch = loader.prelaunch_0[:]
            # INFO: Without the postlaunch coefficient the RMSE go to 15%
            postlaunch = loader.postlaunch[:]
            normalized_data = (np.float32(raw_data) / counts_shift -
                               space_measurement)
            self._cached_calibrated_data = (normalized_data
                                            * postlaunch
                                            * prelaunch)
        return self._cached_calibrated_data

    def getalbedo(self, radiance, totalirradiance, excentricity, zenithangle):
        zenithangle = np.deg2rad(zenithangle)
        return (np.pi * radiance) / (totalirradiance * excentricity
                                     * np.cos(zenithangle))

    def getsatellitalzenithangle(self, lat, lon, sub_lon):
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
        return np.rad2deg(np.pi - np.arccos((h ** 2 - re ** 2 - rs ** 2) /
                                            (-2 * re * rs)))

    def getcorrectedelevation(self, elevation):
        elevation = np.deg2rad(elevation)
        return np.rad2deg(elevation
                          + 0.061359 * ((0.1594 + 1.1230 * elevation +
                                         0.065656 * np.power(elevation, 2)) /
                                        (1 + 28.9344 * elevation +
                                         277.3971 * np.power(elevation, 2))))

    def gethorizontalirradiance(self, extraterrestrialirradiance, excentricity,
                                zenitangle):
        zenitangle = np.deg2rad(zenitangle)
        return extraterrestrialirradiance * excentricity * np.cos(zenitangle)

    def getzenithdiffusetransmitance(self, linketurbidity):
        return (-0.015843 + 0.030543 * linketurbidity +
                0.0003797 * np.power(linketurbidity, 2))

    def getangularcorrection(self, solarelevation, linketurbidity):
        solarelevation = np.deg2rad(solarelevation)
        a0 = (0.264631 - 0.061581 * linketurbidity +
              0.0031408 * np.power(linketurbidity, 2))
        a1 = (2.0402 + 0.018945 * linketurbidity -
              0.011161 * np.power(linketurbidity, 2))
        a2 = (-1.3025 + 0.039231 * linketurbidity +
              0.0085079 * np.power(linketurbidity, 2))
        zenitdiffusetransmitance = self.getzenithdiffusetransmitance(
            linketurbidity)
        c = a0*zenitdiffusetransmitance < 0.002
        a0[c] = 0.002 / zenitdiffusetransmitance[c]
        return (a0 + a1 * np.sin(solarelevation)
                + a2 * np.power(np.sin(solarelevation), 2))

    def getopticalpath(self, correctedelevation, terrainheight,
                       atmosphere_theoretical_height):
        # It set all the negative correctedelevation's values to zero to
        # avoid the use of complex numbers.
        correctedelevation[correctedelevation < 0] = 0
        # In the next line the correctedelevation is used over a degree base.
        power = np.power(correctedelevation + 6.07995,
                         -1.6364)
        # Then should be used over a radian base.
        correctedelevation = np.deg2rad(correctedelevation)
        return (np.exp(-terrainheight/atmosphere_theoretical_height) /
                (np.sin(correctedelevation) + 0.50572 * power))

    def getopticaldepth(self, opticalpath):
        tmp = np.zeros(opticalpath.shape) + 1.0
        highslopebeam = opticalpath <= 20
        lowslopebeam = opticalpath > 20
        opticalpath_power = lambda p: np.power(opticalpath[highslopebeam], p)
        tmp[highslopebeam] = (6.6296 + 1.7513 * opticalpath[highslopebeam] -
                              0.1202 * opticalpath_power(2) +
                              0.0065 * opticalpath_power(3) -
                              0.00013 * opticalpath_power(4))
        tmp[highslopebeam] = 1 / tmp[highslopebeam]
        tmp[lowslopebeam] = 1 / (10.4 + 0.718 * opticalpath[lowslopebeam])
        return tmp

    def getbeamtransmission(self, linketurbidity, opticalpath, opticaldepth):
        return np.exp(-0.8662 * linketurbidity * opticalpath * opticaldepth)

    def getdiffusetransmitance(self, linketurbidity, solarelevation):
        return (self.getzenithdiffusetransmitance(linketurbidity) *
                self.getangularcorrection(solarelevation, linketurbidity))

    def getdiffuseirradiance(self, extraterrestrialirradiance, excentricity,
                             solarelevation, linketurbidity):
        return (extraterrestrialirradiance * excentricity *
                self.getdiffusetransmitance(linketurbidity, solarelevation))

    def gettransmitance(self, linketurbidity, opticalpath, opticaldepth,
                        solarelevation):
        return (self.getbeamtransmission(linketurbidity, opticalpath,
                                         opticaldepth) +
                self.getdiffusetransmitance(linketurbidity, solarelevation))

    def getbeamirradiance(self, extraterrestrialirradiance, excentricity,
                          zenitangle,
                          solarelevation, linketurbidity, terrainheight):
        correctedsolarelevation = self.getcorrectedelevation(solarelevation)
        # TODO: Meteosat is at 8434.5 mts
        opticalpath = self.getopticalpath(correctedsolarelevation,
                                          terrainheight, 8434.5)
        opticaldepth = self.getopticaldepth(opticalpath)
        return (self.gethorizontalirradiance(extraterrestrialirradiance,
                                             excentricity, zenitangle)
                * self.getbeamtransmission(linketurbidity, opticalpath,
                                           opticaldepth))

    def getdailyangle(self, julianday, totaldays):
        return np.rad2deg(2 * np.pi * (julianday - 1) / totaldays)

    def getelevation(self, zenithangle):
        zenithangle = np.deg2rad(zenithangle)
        return np.rad2deg((np.pi / 2) - zenithangle)

    def getglobalirradiance(self, beamirradiance, diffuseirradiance):
        return beamirradiance + diffuseirradiance

    def getatmosphericradiance(self, extraterrestrialirradiance, i0met,
                               diffuseclearsky, satellitalzenitangle):
        satellitalzenitangle = np.deg2rad(satellitalzenitangle)
        anglerelation = np.power((0.5 / np.cos(satellitalzenitangle)), 0.8)
        return ((i0met * diffuseclearsky * anglerelation)
                / (np.pi * extraterrestrialirradiance))

    def getdifferentialalbedo(self, firstalbedo, secondalbedo, t_earth, t_sat):
        return (firstalbedo - secondalbedo) / (t_earth * t_sat)

    def getapparentalbedo(self, observedalbedo, atmosphericalbedo, t_earth,
                          t_sat):
        apparentalbedo = self.getdifferentialalbedo(observedalbedo,
                                                    atmosphericalbedo,
                                                    t_earth, t_sat)
        apparentalbedo[apparentalbedo < 0] = 0.0
        return apparentalbedo

    def geteffectivealbedo(self, solarangle):
        solarangle = np.deg2rad(solarangle)
        return 0.78 - 0.13 * (1 - np.exp(-4 * np.power(np.cos(solarangle), 5)))

    def getcloudalbedo(self, effectivealbedo, atmosphericalbedo,
                       t_earth, t_sat):
        cloudalbedo = self.getdifferentialalbedo(effectivealbedo,
                                                 atmosphericalbedo,
                                                 t_earth, t_sat)
        cloudalbedo[cloudalbedo < 0.2] = 0.2
        effectiveproportion = 2.24 * effectivealbedo
        condition = cloudalbedo > effectiveproportion
        cloudalbedo[condition] = effectiveproportion[condition]
        return cloudalbedo

    def calculate_temporaldata(self, static):
        lat, lon = static.lat, static.lon
        self.declination = self.getdeclination(self.gamma)
        hourlyangle = self.gethourlyangle(lat, lon,
                                          self.decimalhour,
                                          self.gamma)
        self.solarangle = self.getzenithangle(self.declination,
                                              lat,
                                              hourlyangle)
        self.solarelevation = self.getelevation(self.solarangle)
        self.excentricity = self.getexcentricity(self.gamma)
        linke = np.vstack([map(lambda m: static.linke[0, m[0][0] - 1, :],
                               self.months.tolist())])
        # The average extraterrestrial irradiance is 1367.0 Watts/meter^2
        # The maximum height of the non-transparent atmosphere is at 8434.5 mts
        bc = self.getbeamirradiance(1367.0, self.excentricity,
                                    self.solarangle, self.solarelevation,
                                    linke, static.dem)
        dc = self.getdiffuseirradiance(1367.0, self.excentricity,
                                       self.solarelevation, linke)
        self.gc = self.getglobalirradiance(bc, dc)
        satellitalzenithangle = self.getsatellitalzenithangle(
            lat, lon, self.algorithm.SAT_LON)
        atmosphericradiance = self.getatmosphericradiance(
            1367.0, self.algorithm.i0met, dc, satellitalzenithangle)
        self.atmosphericalbedo = self.getalbedo(atmosphericradiance,
                                                self.algorithm.i0met,
                                                self.excentricity,
                                                satellitalzenithangle)
        satellitalelevation = self.getelevation(satellitalzenithangle)
        satellital_opticalpath = self.getopticalpath(
            self.getcorrectedelevation(satellitalelevation),
            static.dem, 8434.5)
        satellital_opticaldepth = self.getopticaldepth(satellital_opticalpath)
        self.t_sat = self.gettransmitance(linke, satellital_opticalpath,
                                          satellital_opticaldepth,
                                          satellitalelevation)
        solar_opticalpath = self.getopticalpath(
            self.getcorrectedelevation(self.solarelevation),
            static.dem, 8434.5)
        solar_opticaldepth = self.getopticaldepth(solar_opticalpath)
        self.t_earth = self.gettransmitance(linke, solar_opticalpath,
                                            solar_opticaldepth,
                                            self.solarelevation)
        effectivealbedo = self.geteffectivealbedo(self.solarangle)
        self.cloudalbedo = self.getcloudalbedo(effectivealbedo,
                                               self.atmosphericalbedo,
                                               self.t_earth,
                                               self.t_sat)

    def getsecondmin(self, albedo):
        min1_albedo = np.ma.masked_array(albedo,
                                         albedo == np.amin(albedo, axis=0))
        return np.amin(min1_albedo, axis=0)

    def getsolarelevation(self, declination, lat, omega):
        omega = np.deg2rad(omega)
        declination = np.deg2rad(declination)
        lat = np.deg2rad(lat)
        return np.rad2deg(np.arcsin(np.sin(declination) * np.sin(lat) +
                                    np.cos(declination) * np.sin(lat) *
                                    np.cos(omega)))

    def getcloudindex(self, apparentalbedo, groundalbedo, cloudalbedo):
        return (apparentalbedo - groundalbedo) / (cloudalbedo - groundalbedo)

    def getclearsky(self, cloudindex):
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

    def calculate_imagedata(self, static, loader, output):
        excentricity = self.excentricity
        solarangle = self.solarangle
        atmosphericalbedo = self.atmosphericalbedo
        t_earth = self.t_earth
        t_sat = self.t_sat
        observedalbedo = self.getalbedo(self.getcalibrateddata(loader),
                                        self.algorithm.i0met,
                                        excentricity, solarangle)
        apparentalbedo = self.getapparentalbedo(observedalbedo,
                                                atmosphericalbedo,
                                                t_earth, t_sat)
        declination = self.declination
        logging.info("Calculating the noon window... ")
        slot_window_in_hours = 4
        image_per_day = 24 * self.algorithm.IMAGE_PER_HOUR
        noon_slot = image_per_day / 2
        half_window = self.algorithm.IMAGE_PER_HOUR * slot_window_in_hours/2
        min_slot = noon_slot - half_window
        max_slot = noon_slot + half_window
        condition = ((self.slots >= min_slot) & (self.slots < max_slot))
        condition = np.reshape(condition, condition.shape[0])
        mask1 = (self.getcalibrateddata(loader)[condition] <=
                 (self.algorithm.i0met / np.pi) * 0.03)
        m_apparentalbedo = np.ma.masked_array(apparentalbedo[condition], mask1)
        # To do the nexts steps needs a lot of memory
        logging.info("Calculating the ground reference albedo... ")
        mask2 = m_apparentalbedo < stats.scoreatpercentile(m_apparentalbedo, 5)
        p5_apparentalbedo = np.ma.masked_array(m_apparentalbedo, mask2)
        groundreferencealbedo = self.getsecondmin(p5_apparentalbedo)
        # Calculate the solar elevation using times, latitudes and omega
        logging.info("Calculating solar elevation... ")
        r_alphanoon = self.getsolarelevation(declination, static.lat, 0)
        r_alphanoon = r_alphanoon * 2./3.
        r_alphanoon[r_alphanoon > 40] = 40
        r_alphanoon[r_alphanoon < 15] = 15
        solarelevation = self.solarelevation
        logging.info("Calculating the ground minimum albedo... ")
        groundminimumalbedo = self.getsecondmin(
            np.ma.masked_array(
                apparentalbedo[condition],
                solarelevation[condition] < r_alphanoon[condition]))
        aux_2g0 = 2 * groundreferencealbedo
        aux_05g0 = 0.5 * groundreferencealbedo
        condition_2g0 = groundminimumalbedo > aux_2g0
        condition_05g0 = groundminimumalbedo < aux_05g0
        groundminimumalbedo[condition_2g0] = aux_2g0[condition_2g0]
        groundminimumalbedo[condition_05g0] = aux_05g0[condition_05g0]
        logging.info("Calculating the cloud index... ")
        cloudindex = self.getcloudindex(apparentalbedo,
                                        groundminimumalbedo,
                                        self.cloudalbedo)
        output.ref_cloudindex[:] = cloudindex
        output.ref_globalradiation[:] = (self.getclearsky(cloudindex) *
                                         self.gc)


strategy = CPUStrategy
