import numpy as np
from netcdf import netcdf as nc
import stats
from helpers import show
from models.core import gpuarray, cuda, SourceModule
from cpu import CPUStrategy, GREENWICH_LON
import itertools


mod_sourcecode = SourceModule(
    """
    #include <stdio.h>

    #define GREENWICH_LON (""" + str(GREENWICH_LON) + """f)
    #define PI (""" + str(np.pi) + """f)
    #define DEG2RAD (float) (PI / 180.0f)
    #define RAD2DEG (float) (180.0f / PI)

    #define i_dt (threadIdx.z)
    #define i_dxy (blockIdx.x + (blockIdx.y * gridDim.x))
    #define i_dxyt (i_dxy + gridDim.x * gridDim.y * i_dt)

    __device__ void getdeclination(float *declination, float *gamma)
    {
        const float g = gamma[i_dt] * DEG2RAD;
        declination[i_dt] = round((0.006918f - 0.399912f * cos(g) +
                        0.070257f * sin(g) -
                        0.006758f * cos(2.0f * g) +
                        0.000907f * sin(2.0f * g) -
                        0.002697f * cos(3.0f * g) +
                        0.00148f * sin(3.0f * g)) * RAD2DEG);
    }

    __device__ float gethourlyangle(float *lat, float *lon,
    float *decimalhour, float *gamma)
    {
        const float g = gamma[i_dt] * DEG2RAD;
        float timeequation = (0.000075f + 0.001868f * cos(g) -
            0.032077f * sin(g) -
            0.014615f * cos(2.0f * g) -
            0.04089f * sin(2.0f * g)) * (12.0f / PI);
        float lon_diff = (GREENWICH_LON - lon[i_dxy]) * DEG2RAD;
        float tst_hour = decimalhour[i_dt] - lon_diff * (12.0f / PI) +
            timeequation;
        float lat_sign = lat[i_dxy] / abs(lat[i_dxy]);
        return ((tst_hour - 12.0f) * lat_sign * PI / 12.0f) * RAD2DEG;
    }

    __device__ void getzenithangle(float *solarangle, float *declination,
    float *lat, float *lon, float *decimalhour, float *gamma)
    {
        float hourlyangle = 0;
        hourlyangle = gethourlyangle(lat, lon, decimalhour, gamma) * DEG2RAD;
        float lat_r = lat[i_dxy] * DEG2RAD;
        float dec_r = declination[i_dt] * DEG2RAD;
        solarangle[i_dxyt] = acos(sin(dec_r) * sin(lat_r) + cos(dec_r) *
            cos(lat_r) * cos(hourlyangle)) * RAD2DEG;
    }

    __device__ float getelevation(float zenithangle)
    {
        float za = zenithangle * DEG2RAD;
        return ((PI / 2.0f) - za) * RAD2DEG;
    }

    __device__ void getexcentricity(float *result, float *gamma)
    {
        const float g = gamma[i_dt] * DEG2RAD;
        result[i_dxyt] = 1.000110f + 0.034221f * cos(g) +
            0.001280f * sin(g) +
            0.000719f * cos(2.0f * g) +
            0.000077f * sin(2.0f * g);
    }

    __device__ float getcorrectedelevation(float elevation)
    {
        float corrected;
        float e = elevation * DEG2RAD;
        float p = pow(e, 2.0f);
        corrected = (e +
                     0.061359f * ((0.1594f + 1.1230f * e +
                                  0.065656f * p) /
                                 (1.0f + 28.9344f * e +
                                  277.3971f * p))) * RAD2DEG;
        return corrected;
    }

    __device__ float getopticalpath(float correctedelevation,
    float *terrainheight, float *atmosphere_theoretical_height)
    {
        float ce = correctedelevation;
        if (ce < 0) { ce = 0.0f; }
        // In the next line the correctedelevation is used over a degree base.
        float p = pow(ce + 6.07995f, -1.6364f);
        ce *= DEG2RAD;
        return (exp(-terrainheight[i_dxy]/atmosphere_theoretical_height[0]) /
               (sin(ce) + 0.50572f * p));
    }


    __device__ float getopticaldepth(float opticalpath)
    {
        float tmp = 1.0f;
        if (opticalpath <= 20.0f){
            tmp = (6.6296f + 1.7513f * opticalpath -
                   0.1202f * pow(opticalpath, 2.0f) +
                   0.0065f * pow(opticalpath, 3.0f) -
                   0.00013f * pow(opticalpath, 4.0f));
        } else {
            tmp = (10.4f + 0.718f * opticalpath);
        }
        tmp = 1.0f / tmp;
        return tmp;
    }

    __device__ float getbeamtransmission(float *linketurbidity,
    float opticalpath, float opticaldepth)
    {
        return exp(-0.8662f * linketurbidity[i_dxyt] * opticalpath *
                   opticaldepth);
    }


    __device__ float gethorizontalirradiance(float *extraterrestrialirradiance,
    float *excentricity, float *zenithangle)
    {
        float radzenith = zenithangle[i_dxyt] * DEG2RAD;
        return extraterrestrialirradiance[0] * excentricity[i_dt] *
               cos(radzenith);
    }


    __device__ float getbeamirradiance(float *extraterrestrialirradiance,
    float *excentricity, float *zenithangle, float solarelevation,
    float *linketurbidity, float *terrainheight)
    {
        float corrected = getcorrectedelevation(solarelevation);
        float opticalpath = getopticalpath(corrected, terrainheight,
                                           extraterrestrialirradiance);
        float opticaldepth = getopticaldepth(opticalpath);
        return gethorizontalirradiance(extraterrestrialirradiance,
                                       excentricity, zenithangle) *
               getbeamtransmission(linketurbidity, opticalpath,
                                   opticaldepth);
    }

    __device__ float getzenithdiffusetransmitance(float *linketurbidity)
    {
        return -0.015843f + 0.030543f * linketurbidity[i_dxyt] +
               0.0003797f * pow(linketurbidity[i_dxyt], 2.0f);
    }

    __device__ float getangularcorrection(float solarelevation,
    float *linketurbidity)
    {
        float sin_se = sin(solarelevation * DEG2RAD);
        float a0 = 0.264631f - 0.061581f * linketurbidity[i_dxyt] +
                   0.0031408f * pow(linketurbidity[i_dxyt], 2.0f);
        float a1 = 2.0402f + 0.018945f * linketurbidity[i_dxyt] -
                   0.011161f * pow(linketurbidity[i_dxyt], 2.0f);
        float a2 = -1.3025f + 0.039231f * linketurbidity[i_dxyt] +
                   0.0085079f * pow(linketurbidity[i_dxyt], 2.0f);
        float ztdifftr = getzenithdiffusetransmitance(linketurbidity);
        if (a0 * ztdifftr < 0.002f){
           a0 = 0.002f / ztdifftr;
        }
        return a0 + a1 * sin_se + a2 * pow(sin_se, 2.0f);
    }

    __device__ float getdiffusetransmitance(float *linketurbidity,
    float solarelevation)
    {
        return getzenithdiffusetransmitance(linketurbidity) *
               getangularcorrection(solarelevation, linketurbidity);
    }

    __device__ void gettransmitance(float *result, float *linketurbidity,
    float opticalpath, float opticaldepth, float solarelevation)
    {
        result[i_dxyt] = getbeamtransmission(linketurbidity, opticalpath,
                   opticaldepth) +
                   getdiffusetransmitance(linketurbidity, solarelevation);
    }

    __device__ float getdiffuseirradiance(float *extraterrestrialirradiance,
    float *excentricity, float solarelevation, float *linketurbidity)
    {
        return extraterrestrialirradiance[0] * excentricity[i_dt] *
               getdiffusetransmitance(linketurbidity, solarelevation);
    }

    __device__ void getglobalirradiance(float *gc, float beamirradiance,
    float diffuseirradiance)
    {
        gc[i_dxyt] = beamirradiance + diffuseirradiance;
    }


    #define rpol 6356.5838f
    #define req  6378.1690f
    #define h    42166.55637f
    //define h    42164.0f

    __device__ float getsatellitalzenithangle(float *lat,
    float *lon, float *sub_lon)
    {
        float la = lat[i_dxy] * DEG2RAD;
        float lon_diff = (lon[i_dxy] - sub_lon[1]) * DEG2RAD;
        float lat_cos_only = cos(la);
        float re = rpol / (sqrt(1 - (pow(req, 2.0f) - pow(rpol, 2.0f)) /
            (pow(req, 2.0f)) * pow(lat_cos_only, 2.0f)));
        float lat_cos = re * lat_cos_only;
        float r1 = h - lat_cos * cos(lon_diff);
        float r2 = - lat_cos * sin(lon_diff);
        float r3 = re * sin(la);
        float rs = sqrt(pow(r1, 2.0f) + pow(r2, 2.0f) + pow(r3, 2.0f));
        return (PI - acos((pow(h, 2.0f) -
            pow(re, 2.0f) - pow(rs, 2.0f)) / (-2.0f * re * rs))) * RAD2DEG;
    }

    __device__ float getatmosphericradiance(float *extraterrestrialirradiance,
    float *i0met, float diffuseclearsky, float satellitalzenithangle)
    {
        float anglerelation = pow(0.5f / cos(satellitalzenithangle * DEG2RAD),
                                  0.8f);
        return (i0met[0] * diffuseclearsky * anglerelation) /
               (PI * extraterrestrialirradiance[0]);
    }

    __device__ float getdifferentialalbedo(float firstalbedo,
    float secondalbedo, float t_earth, float t_sat)
    {
        return (firstalbedo - secondalbedo) / (t_earth * t_sat);
    }

    __device__ void getalbedo(float *result, float radiance,
    float *totalirradiance, float *excentricity, float zenithangle)
    {
        result[i_dxyt] = (PI * radiance) /
                         (totalirradiance[0] * excentricity[i_dt] *
                         cos(zenithangle * DEG2RAD));
    }

    __device__ float geteffectivealbedo(float solarangle)
    {
        return 0.78f - 0.13f * (1.0f -
            exp(-4.0f * pow(cos(solarangle * DEG2RAD), 5.0f)));
    }


    __device__ void getcloudalbedo(float *result, float effectivealbedo,
    float atmosphericalbedo, float t_earth, float t_sat)
    {
        float ca = getdifferentialalbedo(effectivealbedo, atmosphericalbedo,
                            t_earth, t_sat);
        if (ca < 0.2f) { ca = 0.2f; }
        float effectiveproportion = 2.24f * effectivealbedo;
        if ( ca > effectiveproportion) { ca = effectiveproportion; }
        result[i_dxyt] = ca;
    }

    __global__ void update_temporalcache(float *declination,
    float *solarangle, float *solarelevation, float *excentricity,
    float *gc, float *atmosphericalbedo, float *t_sat, float *t_earth,
    float *cloudalbedo, float *lat, float *lon, float *decimalhour,
    float *gamma, float *dem, float *linke, float *SAT_LON,
    float *i0met, float *EXT_RAD, float *HEIGHT)
    {
        float bc, dc, satellitalzenithangle, atmosphericradiance,
              satellitalelevation, satellital_opticalpath,
              satellital_opticaldepth, solar_opticalpath, solar_opticaldepth,
              effectivealbedo;
        getdeclination(declination, gamma);
        getzenithangle(solarangle, declination, lat, lon, decimalhour, gamma);
        solarelevation[i_dxyt] = getelevation(solarangle[i_dxyt]);
        getexcentricity(excentricity, gamma);
        bc = getbeamirradiance(EXT_RAD, excentricity, solarangle,
                               solarelevation[i_dxyt], linke, dem);
        dc = getdiffuseirradiance(EXT_RAD, excentricity,
                                solarelevation[i_dxyt], linke);
        getglobalirradiance(gc, bc, dc);
        satellitalzenithangle = getsatellitalzenithangle(lat, lon, SAT_LON);
        atmosphericradiance = getatmosphericradiance(EXT_RAD, i0met,
                                                     dc,
                                                     satellitalzenithangle);
        getalbedo(atmosphericalbedo, atmosphericradiance, i0met,
                  excentricity, satellitalzenithangle);
        satellitalelevation = getelevation(satellitalzenithangle);
        satellital_opticalpath = getopticalpath(
                getcorrectedelevation(solarelevation[i_dxyt]), dem, HEIGHT);
        satellital_opticaldepth = getopticaldepth(satellital_opticalpath);
        gettransmitance(t_sat, linke, satellital_opticalpath,
                        satellital_opticaldepth, satellitalelevation);
        solar_opticalpath = getopticalpath(
                        getcorrectedelevation(solarelevation[i_dxyt]), dem,
                        HEIGHT);
        solar_opticaldepth = getopticaldepth(solar_opticalpath);
        gettransmitance(t_earth, linke, solar_opticalpath, solar_opticaldepth,
                        solarelevation[i_dxyt]);
        effectivealbedo = geteffectivealbedo(solarangle[i_dxyt]);
        getcloudalbedo(cloudalbedo, effectivealbedo,
                          atmosphericalbedo[i_dxyt], t_earth[i_dxyt],
                          t_sat[i_dxyt]);
    }
    """)


def gpu_exec(func_name, results, *matrixs):
    func = mod_sourcecode.get_function(func_name)
    is_num = lambda x: isinstance(x, (int, long, float, complex))
    adapt_matrix = lambda m: m if isinstance(m, np.ndarray) else m[:]
    adapt = lambda x: np.array([[[x]]]) if is_num(x) else adapt_matrix(x)
    matrixs_ram = map(lambda m: adapt(m).astype(np.float32,
                                                casting='same_kind'),
                      matrixs)
    matrixs_gpu = map(lambda m: cuda.mem_alloc(m.nbytes), matrixs_ram)
    transferences = zip(matrixs_ram, matrixs_gpu)
    list(map(lambda (m, m_gpu): cuda.memcpy_htod(m_gpu, m), transferences))
    m_shapes = map(lambda m: list(m.shape), matrixs_ram)
    for m_s in m_shapes:
        while len(m_s) < 3:
            m_s.insert(0, 1)
    blocks = map(lambda ms: ms[1:3], m_shapes)
    size = lambda m: m[0] * m[1]
    max_blocks = max(map(size, blocks))
    blocks = list(reversed(filter(lambda ms: size(ms) == max_blocks, blocks)[0]))
    threads = max(map(lambda ms: ms[0], m_shapes))
    show('-> block by grid: %s, threads by block: %s\n' % (str(blocks), str(threads)))
    func(*matrixs_gpu, grid=tuple(blocks), block=tuple([1, 1, threads]))
    list(map(lambda (m, m_gpu): cuda.memcpy_dtoh(m, m_gpu), transferences[:results]))
    for i in range(results):
        matrixs[i][:] = matrixs_ram[i]
        matrixs_gpu[i].free()
    return matrixs_ram[:results]


class GPUStrategy(CPUStrategy):

    def update_temporalcache(self, loader, cache):
        const = lambda c: np.array(c).reshape(1, 1, 1)
        inputs = [loader.lat,
                  loader.lon,
                  self.decimalhour,
                  self.gamma,
                  loader.dem,
                  loader.linke,
                  const(self.algorithm.SAT_LON),
                  const(self.algorithm.i0met),
                  const(1367.0),
                  const(8434.5)]
        outputs = [self.declination,
                   self.solarangle,
                   self.solarelevation,
                   self.excentricity,
                   self.gc,
                   self.atmosphericalbedo,
                   self.t_sat,
                   self.t_earth,
                   self.cloudalbedo]
        matrixs = list(itertools.chain(*[outputs, inputs]))
        gpu_exec("update_temporalcache", len(outputs),
                 *matrixs)
        print "----"
        maxmin = map(lambda o: (o[:].min(), o[:].max()), outputs)
        for mm in zip(range(len(maxmin)), maxmin):
            print mm[0], ': ', mm[1]
        print "----"
        nc.sync(cache)
        super(GPUStrategy, self).update_temporalcache(loader, cache)


strategy = GPUStrategy
