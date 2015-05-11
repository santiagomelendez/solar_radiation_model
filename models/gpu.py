import numpy as np
from datetime import datetime
from netcdf import netcdf as nc
import stats
from helpers import show
from models.core import gpuarray, cuda, SourceModule
from cpu import CPUStrategy, GREENWICH_LON
import itertools


mod_sourcecode = SourceModule(
    """
    #define GREENWICH_LON """ + str(GREENWICH_LON) + """
    #define PI """ + str(np.pi) + """
    #define DEG2RAD (PI / 180)
    #define RAD2DEG (180 / PI)

    #define i_dt threadIdx.z
    #define i_dxy blockIdx.x + (blockIdx.y * gridDim.x)
    #define i_dxyt (i_dxy) + (gridDim.x * gridDim.y * i_dt)

    __device__ void getdeclination(float *declination, float *gamma)
    {
	float g = gamma[i_dt] * DEG2RAD;
        declination[i_dt] = (0.006918f - 0.399912f * cos(g) +
                        0.070257f * sin(g) -
                        0.006758f * cos(2 * g) +
                        0.000907f * sin(2 * g) -
                        0.002697f * cos(3 * g) +
                        0.00148f * sin(3 * g)) * RAD2DEG;

    }

    __device__ float gethourlyangle(float *lat, float *lon,
    float *decimalhour, float *gamma)
    {
	float g = gamma[i_dt] * DEG2RAD;
        float timeequation = (0.000075 + 0.001868 * cos(g) -
            0.032077 * sin(g) -
            0.014615 * cos(2 * g) -
            0.04089 * sin(2 * g)) * (12 / PI);
        float lon_diff = (GREENWICH_LON - lon[i_dxy]) * DEG2RAD;
        float tst_hour = decimalhour[i_dt] - lon_diff * (12 / PI) + timeequation;
        float lat_sign = lat[i_dxy] / abs(lat[i_dxy]);
        return ((tst_hour - 12) * lat_sign * PI / 12) * RAD2DEG;
    }

    __device__ void getzenithangle(float *solarangle, float *declination,
    float *lat, float *lon, float *decimalhour, float *gamma)
    {
	float hourlyangle;
	hourlyangle = gethourlyangle(lat, lon, decimalhour, gamma);
        hourlyangle *= DEG2RAD;
        float lat_r = lat[i_dxy] * DEG2RAD;
        float dec_r = declination[i_dt] * DEG2RAD;
        solarangle[i_dxyt] = acos(sin(dec_r) * sin(lat_r) + cos(dec_r) *
            cos(lat_r) * cos(hourlyangle)) * RAD2DEG;
    }

    __global__ void getexcentricity(float *result, float *gamma)
    {
        gamma[i_dt] *= DEG2RAD;
        result[i_dt] = 1.000110 + 0.034221 * cos(gamma[i_dt]) + \
            0.001280 * sin(gamma[i_dt]) + \
            0.000719 * cos(2 * gamma[i_dt]) + \
            0.000077 * sin(2 * gamma[i_dt]);
    }


    __global__ void getsatellitalzenthangle(float *result, float *lat, \
    float *lon, float sub_lon, float rpol, float req, float h)
    {
        lat[i_dxy] *= DEG2RAD;
        float lon_diff = (lon[i_dxy] - sub_lon) * RAD2DEG;
        float lat_cos_only = cos(lat[i_dxy]);
        float re = rpol / (sqrt(1 - (pow(req, 2) - pow(rpol, 2)) / \
            (pow(req, 2)) * pow(lat_cos_only, 2)));
        float lat_cos = re * lat_cos_only;
        float r1 = h - lat_cos * cos(lon_diff);
        float r2 = - lat_cos * sin(lon_diff);
        float r3 = re * sin(lat[i_dxy]);
        float rs = sqrt(pow(r1,2) + pow(r2,2) + pow(r3,2));
        result[i_dxy] = (PI - acos((pow(h,2) - \
            pow(re, 2) - pow(rs, 2)) / (-2 * re * rs)));
        result[i_dxy] *= RAD2DEG;
    }

    __global__ void getalbedo(float *result, float *radiance, \
    float totalirradiance, float *excentricity, float *zenithangle)
    {
        result[i_dt] = (PI * \
            radiance[i_dt]) / (totalirradiance * excentricity[i_dt] * \
            cos(zenithangle[i_dt]));
    }

    __global__ void update_temporalcache(float *declination,
    float *solarelevation, float* solarangle, float *excentricity,
    float *gc, float *atmosphericalbedo, float *t_sat, float *t_earth,
    float *cloudalbedo, float *lat, float *lon, float *times,
    float *decimalhour,  float *gamma, float *dem, float *linke,
    float *SAT_LON, float *i0met, float *EXT_RAD, float *HEIGHT)
    {
        getdeclination(declination, gamma);
        getzenithangle(solarangle, declination, lat, lon, decimalhour, gamma);
    }
    """)


def gpu_exec(func_name, results, *matrixs):
    func = mod_sourcecode.get_function(func_name)
    is_num = lambda x: isinstance(x, (int, long, float, complex))
    adapt_matrix = lambda m: m if isinstance(m, np.ndarray) else m[:]
    adapt = lambda x: np.array([[[x]]]) if is_num(x) else adapt_matrix(x)
    matrixs_ram = map(lambda m: adapt(m).astype(np.float32), matrixs)
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


def getexcentricity(gamma):
    result = np.empty_like(gamma)
    gpu_exec("getexcentricity", 1, result, gamma)
    return result


def getzenithangle(declination, latitude, hourlyangle):
    result = np.empty_like(declination)
    gpu_exec("getzenithangle", 1, result, hourlyangle, latitude, declination)
    return result


def getalbedo(radiance, totalirradiance, excentricity, zenitangle):
    result = np.empty_like(radiance)
    gpu_exec("getalbedo", 1, result, radiance, totalirradiance,
             excentricity, zenitangle)
    return result


def getsatellitalzenithangle(lat, lon, sub_lon):
    rpol = 6356.5838
    req = 6378.1690
    h = 42166.55637  # 42164.0
    result = np.empty_like(lat)
    gpu_exec("getsatellitalzenithangle", 1, result,
             lat, lon, sub_lon, rpol, req, h)
    return result


class GPUStrategy(CPUStrategy):

    """
    def getexcentricity(self, gamma):
        return getexcentricity(gamma)
    """

    def getdeclination(self, gamma):
        return self.declination[:]  # getdeclination(gamma)

    def gethourlyangle(self, lat, lon, decimalhour, gamma):
        # TODO: Continue from here!
        tmp = super(GPUStrategy, self).gethourlyangle(lat, lon, decimalhour,
                                                      gamma)
        print tmp, self.solarelevation[:]
        return self.solarelevation[:]

    """
    def getzenithangle(self, declination, latitude, hourlyangle):
        return getzenithangle(declination, latitude, hourlyangle)

    def getalbedo(self, radiance, totalirradiance, excentricity, zenitangle):
        return getalbedo(radiance, totalirradiance, excentricity,
                         zenitangle)

    def getsatellitalzenithangle(self, lat, lon, sub_lon):
        return getsatellitalzenithangle(lat, lon, sub_lon)
    """

    def update_temporalcache(self, loader, cache):
        const = lambda c: np.array(c).reshape(1, 1, 1)
        inputs = [loader.lat,
                  loader.lon,
                  self.times,
                  self.decimalhour,
                  self.gamma,
                  loader.dem,
                  loader.linke,
                  const(self.algorithm.SAT_LON),
                  const(self.algorithm.i0met),
                  const(1367.0),
                  const(8434.5)]
        outputs = [self.declination,
                   self.solarelevation,
                   self.solarangle,
                   self.excentricity,
                   self.gc,
                   self.atmosphericalbedo,
                   self.t_sat,
                   self.t_earth,
                   self.cloudalbedo]
        matrixs = list(itertools.chain(*[outputs, inputs]))
        gpu_exec("update_temporalcache", len(outputs),
                 *matrixs)
        nc.sync(cache)
        super(GPUStrategy, self).update_temporalcache(loader, cache)


strategy = GPUStrategy
