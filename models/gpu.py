import numpy as np
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
    #define i_dxy (blockIdx.x + (blockIdx.y * gridDim.x))
    #define i_dxyt (i_dxy + gridDim.x * gridDim.y * i_dt)

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
        float tst_hour = decimalhour[i_dt] - lon_diff * (12 / PI) +
            timeequation;
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

    __device__ void getelevation(float *solarelevation, float *zenithangle)
    {
        float za = zenithangle[i_dxyt] * DEG2RAD;
        solarelevation[i_dxyt] = ((PI / 2) - za) * RAD2DEG;
    }

    __device__ void getexcentricity(float *result, float *gamma)
    {
        float g = gamma[i_dt] * DEG2RAD;
        result[i_dxyt] = 1.000110 + 0.034221 * cos(g) +
            0.001280 * sin(g) +
            0.000719 * cos(2 * g) +
            0.000077 * sin(2 * g);
    }

    __device__ float getcorrectedelevation(float *elevation)
    {
        float corrected;
        float e = elevation[i_dxyt] * DEG2RAD;
        float p = pow(e, 2);
        corrected = (e +
                     0.061359 * ((0.1594 + 1.1230 * e +
                                  0.065656 * p) /
                                 (1 + 28.9344 * e +
                                  277.3971 * p))) * RAD2DEG;
        return corrected;
    }

    __device__ float getopticalpath(float correctedelevation,
    float *terrainheight, float *atmosphere_theoretical_height)
    {
        float ce = correctedelevation;
        if (ce < 0) { ce = 0.0f; }
        // In the next line the correctedelevation is used over a degree base.
        float p = pow(ce + 6.07995, -1.6364);
        ce *= DEG2RAD;
        return (exp(-terrainheight[i_dxy]/atmosphere_theoretical_height[0]) /
               (sin(ce) + 0.50572 * p));
    }


    __device__ float getopticaldepth(float opticalpath)
    {
        float tmp = 1.0f;
        if (opticalpath <= 20){
            tmp = (6.6296 + 1.7513 * opticalpath -
                   0.1202 * pow(opticalpath, 2) +
                   0.0065 * pow(opticalpath, 3) -
                   0.00013 * pow(opticalpath, 4));
        } else {
            tmp = (10.4 + 0.718 * opticalpath);
        }
        tmp = 1 / tmp;
        return tmp;
    }

    __device__ float getbeamtransmission(float *linketurbidity,
    float opticalpath, float opticaldepth)
    {
        return exp(-0.8662 * linketurbidity[i_dxyt] * opticalpath *
                   opticaldepth);
    }


    __device__ float gethorizontalirradiance(float *extraterrestrialirradiance,
    float *excentricity, float *zenithangle)
    {
        float radzenith = zenithangle[i_dxyt] * DEG2RAD;
        return extraterrestrialirradiance[0] * excentricity[i_dxyt] *
               cos(radzenith);
    }


    __device__ float getbeamirradiance(float *extraterrestrialirradiance,
    float *excentricity, float *zenithangle, float *solarelevation,
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
        return -0.015843 + 0.030543 * linketurbidity[i_dxyt] +
               0.0003797 * pow(linketurbidity[i_dxyt], 2);
    }

    __device__ float getangularcorrection(float *solarelevation,
    float *linketurbidity)
    {
        float sin_se = sin(solarelevation[i_dxyt] * DEG2RAD);
        float a0 = 0.264631 - 0.061581 * linketurbidity[i_dxyt] +
                   0.0031408 * pow(linketurbidity[i_dxyt], 2);
        float a1 = 2.0402 + 0.018945 * linketurbidity[i_dxyt] -
                   0.011161 * pow(linketurbidity[i_dxyt], 2);
        float a2 = -1.3025 + 0.039231 * linketurbidity[i_dxyt] +
                   0.0085079 * pow(linketurbidity[i_dxyt], 2);
        float ztdifftr = getzenithdiffusetransmitance(linketurbidity);
        if (a0 * ztdifftr < 0.002){
           a0 = 0.002 / ztdifftr;
        }
        return a0 + a1 * sin_se + a2 * pow(sin_se, 2);
    }

    __device__ float getdiffusetransmitance(float *linketurbidity,
    float *solarelevation)
    {
        return getzenithdiffusetransmitance(linketurbidity) *
               getangularcorrection(solarelevation, linketurbidity);
    }

    __device__ float getdiffuseirradiance(float *extraterrestrialirradiance,
    float *excentricity, float *solarelevation, float *linketurbidity)
    {
        return extraterrestrialirradiance[0] * excentricity[i_dt] *
               getdiffusetransmitance(linketurbidity, solarelevation);
    }

    __device__ void getglobalirradiance(float *gc, float beamirradiance,
    float diffuseirradiance)
    {
        gc[i_dxyt] = beamirradiance + diffuseirradiance;
    }


    /*
    __global__ void getsatellitalzenithangle(float *result, float *lat,
    float *lon, float sub_lon, float rpol, float req, float h)
    {
        lat[i_dxy] *= DEG2RAD;
        float lon_diff = (lon[i_dxy] - sub_lon) * RAD2DEG;
        float lat_cos_only = cos(lat[i_dxy]);
        float re = rpol / (sqrt(1 - (pow(req, 2) - pow(rpol, 2)) /
            (pow(req, 2)) * pow(lat_cos_only, 2)));
        float lat_cos = re * lat_cos_only;
        float r1 = h - lat_cos * cos(lon_diff);
        float r2 = - lat_cos * sin(lon_diff);
        float r3 = re * sin(lat[i_dxy]);
        float rs = sqrt(pow(r1,2) + pow(r2,2) + pow(r3,2));
        result[i_dxy] = (PI - acos((pow(h,2) -
            pow(re, 2) - pow(rs, 2)) / (-2 * re * rs)));
        result[i_dxy] *= RAD2DEG;
    }

    __global__ void getalbedo(float *result, float *radiance,
    float totalirradiance, float *excentricity, float *zenithangle)
    {
        result[i_dt] = (PI *
            radiance[i_dt]) / (totalirradiance * excentricity[i_dt] *
            cos(zenithangle[i_dt]));
    }
    */


    __global__ void update_temporalcache(float *declination,
    float *solarelevation, float* solarangle, float *excentricity,
    float *gc, float *atmosphericalbedo, float *t_sat, float *t_earth,
    float *cloudalbedo, float *lat, float *lon, float *times,
    float *decimalhour,  float *gamma, float *dem, float *linke,
    float *SAT_LON, float *i0met, float *EXT_RAD, float *HEIGHT)
    {
        float bc, dc;
        getdeclination(declination, gamma);
        getzenithangle(solarangle, declination, lat, lon, decimalhour, gamma);
        getelevation(solarelevation, solarangle);
        getexcentricity(excentricity, gamma);
        bc = getbeamirradiance(EXT_RAD, excentricity, solarangle,
                               solarelevation, linke, dem);
        dc = getdiffuseirradiance(EXT_RAD, excentricity, solarelevation,
                                  linke);
        getglobalirradiance(gc, bc, dc);
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


class GPUStrategy(CPUStrategy):


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
