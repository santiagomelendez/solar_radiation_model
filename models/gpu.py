import numpy as np
from datetime import datetime
from netcdf import netcdf as nc
import stats
from helpers import show
from models.core import gpuarray, cuda, SourceModule
from cpu import CPUStrategy, estimate_globalradiation
import itertools


deg2rad_ratio = str(np.float32(np.pi / 180))
rad2deg_ratio = str(np.float32(180 / np.pi))


mod_sourcecode = SourceModule(
    """
    __global__ void getexcentricity(float *result, float *gamma)
    {
        int it = threadIdx.x;
        int i2d = blockDim.x*blockIdx.x;
        int i3d = i2d + it;
        gamma[i3d] *= """ + deg2rad_ratio + """;
        result[i3d] = 1.000110 + 0.034221 * cos(gamma[i3d]) + \
            0.001280 * sin(gamma[i3d]) + \
            0.000719 * cos(2 * gamma[i3d]) + \
            0.000077 * sin(2 * gamma[i3d]);
    }

    __global__ void getdeclination(float *result, float *gamma)
    {
        int it = threadIdx.x;
        int i2d = blockDim.x * blockIdx.x;
        int i3d = i2d + it;
        gamma[i3d] *= """ + deg2rad_ratio + """;
        result[i3d] = 0.006918 - 0.399912 * cos(gamma[i3d]) + \
            0.070257 * sin(gamma[i3d]) - \
            0.006758 * cos(2 * gamma[i3d]) + \
            0.000907 * sin(2 * gamma[i3d]) - \
            0.002697 * cos(3 * gamma[i3d]) + \
            0.00148 * sin(3 * gamma[i3d]);
        result[i3d] *= """ + rad2deg_ratio + """;
    }

    __global__ void getzenithangle(float *result, float *hourlyangle, float *lat, float *dec)
    {
        int it = threadIdx.x;
        int i2d = blockDim.x * blockIdx.x;
        int i3d = i2d + it;
        float lat_r = lat[i2d] * """ + deg2rad_ratio + """;
        float dec_r = dec[it] * """ + deg2rad_ratio + """;
        hourlyangle[i3d] *= """ + deg2rad_ratio + """;
        result[i3d] = acos(sin(dec_r) * sin(lat_r) + \
            cos(dec_r) * cos(lat_r) * cos(hourlyangle[i3d]));
        result[i3d] *= """ + rad2deg_ratio + """;
    }

    __global__ void getsatellitalzenithangle(float *result, float *lat, \
    float *lon, float sub_lon, float rpol, float req, float h)
    {
        //int it = threadIdx.x;
        int i2d = blockDim.x*blockIdx.x;
        //int i3d = i2d + it;
        lat[i2d] *= """ + deg2rad_ratio + """;
        float lon_diff = (lon[i2d] - sub_lon) * """ + deg2rad_ratio + """;
        float lat_cos_only = cos(lat[i2d]);
        float re = rpol / (sqrt(1 - (pow(req, 2) - pow(rpol, 2)) / \
            (pow(req, 2)) * pow(lat_cos_only, 2)));
        float lat_cos = re * lat_cos_only;
        float r1 = h - lat_cos * cos(lon_diff);
        float r2 = - lat_cos * sin(lon_diff);
        float r3 = re * sin(lat[i2d]);
        float rs = sqrt(pow(r1,2) + pow(r2,2) + pow(r3,2));
        result[i2d] = (""" + str(np.float32(np.pi)) + """ - acos((pow(h,2) - \
            pow(re, 2) - pow(rs, 2)) / (-2 * re * rs)));
        result[i2d] *= """ + rad2deg_ratio + """;
    }

    __global__ void getalbedo(float *result, float *radiance, float totalirradiance, \
    float *excentricity, float *zenitangle)
    {
        int it = threadIdx.x;
        int i2d = blockDim.x*blockIdx.x;
        int i3d = i2d + it;
        result[i3d] = (""" + str(np.float32(np.pi)) + """ * \
            radiance[i3d]) / (totalirradiance * excentricity[i3d] * \
            cos(zenitangle[i3d]));
    }

    __global__ void update_temporalcache(float *declination, \
    float *solarelevation, float* solarangle, float *excentricity, \
    float *gc, float *atmosphericalbedo, float *t_sat, float *t_earth, \
    float *cloudalbedo, float *lan, float *lon, float *times, \
    float *decimalhour,  float *gamma, float *dem, float *linke, \
    float *SAT_LON, float *i0met, float *EXT_RAD, \
    float *HEIGHT)
    {
        const unsigned long long int i1d = blockIdx.x;
        //const unsigned long long int i2d = i1d + gridDim.x * blockIdx.y;
        //const unsigned long long int i3d = i2d + gridDim.x * gridDim.y
        //    * blockIdx.z;
        gamma[i1d] *= """ + deg2rad_ratio + """;
        declination[i1d] = (0.006918 - 0.399912 * cos(gamma[i1d]) +
                        0.070257 * sin(gamma[i1d]) -
                        0.006758 * cos(2 * gamma[i1d]) +
                        0.000907 * sin(2 * gamma[i1d]) -
                        0.002697 * cos(3 * gamma[i1d]) +
                        0.00148 * sin(3 * gamma[i1d]));
        declination[i1d] *= """ + rad2deg_ratio + """;
    }
    """)


def gpu_exec(func_name, results, *matrixs):
    func = mod_sourcecode.get_function(func_name)
    adapt = lambda m: m if isinstance(m, np.ndarray) else np.array([[[m]]])
    matrixs = map(lambda m: adapt(m).astype(np.float32), matrixs)
    matrixs_gpu = map(lambda m: cuda.mem_alloc(m.nbytes), matrixs)
    transferences = zip(matrixs, matrixs_gpu)
    map(lambda (m, m_gpu): cuda.memcpy_htod(m_gpu, m), transferences)
    m_shapes = map(lambda m: list(m.shape), matrixs)
    for m_s in m_shapes:
        while len(m_s) < 3:
            m_s.insert(0, 1)
    # TODO: Verify to work with the complete matrix at the same time.
    func(*matrixs_gpu, grid=tuple(m_shapes[0][1:3]),
         block=tuple([m_shapes[0][0], 1, 1]))
    map(lambda (m, m_gpu): cuda.memcpy_dtoh(m, m_gpu), transferences[:results])
    for m in matrixs_gpu:
        m.free()
    return matrixs[:results]


def getexcentricity(gamma):
    result = np.empty_like(gamma)
    gpu_exec("getexcentricity", 1, result, gamma)
    return result


def getdeclination(gamma):
    result = np.empty_like(gamma)
    gpu_exec("getdeclination", 1, result, gamma)
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

    def getexcentricity(self, gamma):
        return getexcentricity(gamma)

    def getdeclination(self, gamma):
        return self.declination[:]  # getdeclination(gamma)

    def getzenithangle(self, declination, latitude, hourlyangle):
        return getzenithangle(declination, latitude, hourlyangle)

    def getalbedo(self, radiance, totalirradiance, excentricity, zenitangle):
        return getalbedo(radiance, totalirradiance, excentricity,
                         zenitangle)

    def getsatellitalzenithangle(self, lat, lon, sub_lon):
        return getsatellitalzenithangle(lat, lon, sub_lon)

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
        results = [self.declination[:],
                   self.solarelevation[:],
                   self.solarangle[:],
                   self.excentricity[:],
                   self.gc[:],
                   self.atmosphericalbedo[:],
                   self.t_sat[:],
                   self.t_earth[:],
                   self.cloudalbedo[:]]
        matrixs = list(itertools.chain(*[results, inputs]))
        gpu_exec("update_temporalcache", len(results), *matrixs)
        nc.sync(cache)
        return super(GPUStrategy, self).update_temporalcache(loader, cache)


strategy = GPUStrategy
