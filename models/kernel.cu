#include <stdio.h>

#define GREENWICH_LON (0.0f)
#define PI (3.141592653589793f)
#define DEG2RAD (PI / 180.0f)
#define RAD2DEG (180.0f / PI)

#define i_dt (threadIdx.z)
#define i_dxy (threadIdx.x \
               + (blockIdx.x * blockDim.x) \
               + ((threadIdx.y + (blockIdx.y * blockDim.y)) * (gridDim.x * blockDim.x)))
#define s_dxy ((gridDim.x * blockDim.x) * (gridDim.y * blockDim.y))
#define i_dxyt (i_dxy + s_dxy * i_dt)

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

__device__ float getzenithangle(float *declination,
float *lat, float *lon, float *decimalhour, float *gamma)
{
    float hourlyangle;
    hourlyangle = gethourlyangle(lat, lon, decimalhour, gamma) * DEG2RAD;
    float lat_r = lat[i_dxy] * DEG2RAD;
    float dec_r = declination[i_dt] * DEG2RAD;
    return (acos(sin(dec_r) * sin(lat_r) + cos(dec_r) *
        cos(lat_r) * cos(hourlyangle)) * RAD2DEG);
}

__device__ float getelevation(float zenithangle)
{
    float za = zenithangle * DEG2RAD;
    return ((PI / 2.0f) - za) * RAD2DEG;
}

__device__ float getexcentricity(float *gamma)
{
    const float g = gamma[i_dt] * DEG2RAD;
    return (1.000110f + 0.034221f * cos(g) +
        0.001280f * sin(g) +
        0.000719f * cos(2.0f * g) +
        0.000077f * sin(2.0f * g));
}

__device__ float getcorrectedelevation(float elevation)
{
    float e = elevation * DEG2RAD;
    float p = pow(e, 2.0f);
    return (e +
                 0.061359f * ((0.1594f + 1.1230f * e +
                              0.065656f * p) /
                             (1.0f + 28.9344f * e +
                              277.3971f * p))) * RAD2DEG;
}

__device__ float getopticalpath(float correctedelevation,
float *dem, float *HEIGHT)
{
    float ce = correctedelevation;
    if (ce < 0) { ce = 0.0f; }
    // In the next line the correctedelevation is used over a degree base.
    float p = pow(ce + 6.07995f, -1.6364f);
    return(exp(-dem[i_dxy]/HEIGHT[0]) /
           (sin(ce * DEG2RAD) + 0.50572f * p));
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

__device__ float getbeamtransmission(float *linke,
float opticalpath, float opticaldepth)
{
    return exp(-0.8662f * linke[i_dxy] * opticalpath *
               opticaldepth);
}


__device__ float gethorizontalirradiance(float *EXT_RAD,
float *excentricity, float *zenithangle)
{
    float radzenith = zenithangle[i_dxyt] * DEG2RAD;
    return EXT_RAD[0] * excentricity[i_dt] *
           cos(radzenith);
}


__device__ float getbeamirradiance(float *EXT_RAD,
float *excentricity, float *zenithangle, float solarelevation,
float *linke, float *dem, float *HEIGHT)
{
    float corrected = getcorrectedelevation(solarelevation);
    float opticalpath = getopticalpath(corrected, dem,
                                       HEIGHT);
    float opticaldepth = getopticaldepth(opticalpath);
    return gethorizontalirradiance(EXT_RAD,
                                   excentricity, zenithangle) *
           getbeamtransmission(linke, opticalpath,
                               opticaldepth);
}

__device__ float getzenithdiffusetransmitance(float *linke)
{
    return -0.015843f + 0.030543f * linke[i_dxy] +
           0.0003797f * pow(linke[i_dxy], 2.0f);
}

__device__ float getangularcorrection(float solarelevation,
float *linke)
{
    float sin_se = sin(solarelevation * DEG2RAD);
    float squared_linke = pow(linke[i_dxy], 2.0f);
    float a0 = 0.264631f - 0.061581f * linke[i_dxy] +
               0.0031408f * squared_linke;
    float a1 = 2.0402f + 0.018945f * linke[i_dxy] -
               0.011161f * squared_linke;
    float a2 = -1.3025f + 0.039231f * linke[i_dxy] +
               0.0085079f * squared_linke;
    float ztdifftr = getzenithdiffusetransmitance(linke);
    if (a0 * ztdifftr < 0.002f){
       a0 = 0.002f / ztdifftr;
    }
    return a0 + a1 * sin_se + a2 * pow(sin_se, 2.0f);
}

__device__ float getdiffusetransmitance(float *linke,
float solarelevation)
{
    return getzenithdiffusetransmitance(linke) *
           getangularcorrection(solarelevation, linke);
}

__device__ void gettransmitance(float *transmitance, float *linke,
float opticalpath, float opticaldepth, float solarelevation)
{
    transmitance[i_dxyt] = getbeamtransmission(linke, opticalpath,
               opticaldepth) +
               getdiffusetransmitance(linke, solarelevation);
}

__device__ float getdiffuseirradiance(float *EXT_RAD,
float *excentricity, float solarelevation, float *linketurbidity)
{
    return EXT_RAD[0] * excentricity[i_dt] *
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
    float lon_diff = (lon[i_dxy] - sub_lon[0]) * DEG2RAD;
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

__device__ float getatmosphericradiance(float *EXT_RAD,
float *i0met, float diffuseclearsky, float satellitalzenithangle)
{
    float anglerelation = pow(0.5f / cos(satellitalzenithangle * DEG2RAD),
                              0.8f);
    return ((i0met[0] * diffuseclearsky * anglerelation) /
            (PI * EXT_RAD[0]));
}

__device__ float getdifferentialalbedo(float firstalbedo,
float secondalbedo, float t_earth, float t_sat)
{
    return (firstalbedo - secondalbedo) / (t_earth * t_sat);
}

__device__ void getalbedo(float *albedo, float radiance,
float *i0met, float *excentricity, float zenithangle)
{
    albedo[i_dxyt] = ((PI * radiance) /
            (i0met[0] * excentricity[i_dt] *
            cos(zenithangle * DEG2RAD)));
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

__global__ void calculate_temporaldata(float *declination,
float *solarangle, float *solarelevation, float *excentricity,
float *gc, float *atmosphericalbedo, float *t_sat, float *t_earth,
float *cloudalbedo, float *lat, float *lon, float *decimalhour, float *months,
float *gamma, float *dem, float *linke, float *SAT_LON,
float *i0met, float *EXT_RAD, float *HEIGHT)
{
    float bc, dc, satellitalzenithangle, atmosphericradiance,
          satellitalelevation, satellital_opticalpath,
          satellital_opticaldepth, solar_opticalpath,
          solar_opticaldepth, effectivealbedo;
    getdeclination(declination, gamma);
    solarangle[i_dxyt] = getzenithangle(declination, lat, lon,
            decimalhour, gamma);
    solarelevation[i_dxyt] = getelevation(solarangle[i_dxyt]);
    excentricity[i_dt] = getexcentricity(gamma);
    int linke_idx = (months[i_dt] - 1) * s_dxy;
    bc = getbeamirradiance(EXT_RAD, excentricity, solarangle,
                           solarelevation[i_dxyt], linke + linke_idx, dem, HEIGHT);

    dc = getdiffuseirradiance(EXT_RAD, excentricity,
                              solarelevation[i_dxyt], linke + linke_idx);
    getglobalirradiance(gc, bc, dc);
    satellitalzenithangle = getsatellitalzenithangle(lat, lon, SAT_LON);
    atmosphericradiance = getatmosphericradiance(EXT_RAD, i0met,
                                                 dc,
                                                 satellitalzenithangle);
    getalbedo(atmosphericalbedo, atmosphericradiance, i0met,
              excentricity, satellitalzenithangle);
    satellitalelevation = getelevation(satellitalzenithangle);
    satellital_opticalpath = getopticalpath(
            getcorrectedelevation(satellitalelevation), dem, HEIGHT);
    satellital_opticaldepth = getopticaldepth(satellital_opticalpath);
    gettransmitance(t_sat, linke + linke_idx, satellital_opticalpath,
                    satellital_opticaldepth, satellitalelevation);
    solar_opticalpath = getopticalpath(
                    getcorrectedelevation(solarelevation[i_dxyt]), dem,
                    HEIGHT);
    solar_opticaldepth = getopticaldepth(solar_opticalpath);
    gettransmitance(t_earth, linke + linke_idx, solar_opticalpath, solar_opticaldepth,
                    solarelevation[i_dxyt]);
    effectivealbedo = geteffectivealbedo(solarangle[i_dxyt]);
    getcloudalbedo(cloudalbedo, effectivealbedo,
                      atmosphericalbedo[i_dxyt], t_earth[i_dxyt],
                      t_sat[i_dxyt]);
}


__global__ void calculate_imagedata(float *cloudindex,
float *globalradiation, float *slots, float *declination,
float *solarangle, float *solarelevation, float *excentricity, float *lat,
float *calibrated_data, float *gc, float *t_sat, float *t_earth,
float *atmosphericalbedo, float *cloudalbedo,
float *i0met, float *IMAGE_PER_HOUR)
{
}
