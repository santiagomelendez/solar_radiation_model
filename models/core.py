from datetime import datetime
import numpy as np
from netcdf import netcdf as nc
from multiprocessing import Process, Pipe
from itertools import izip
import functools
from cache import memoize


try:
    # raise Exception('Force CPU')
    from pycuda.compiler import SourceModule
    import pycuda.gpuarray as gpuarray
    import pycuda.driver as cuda
    import pycuda.autoinit
    cuda_can_help = True
    print "<< using CUDA cores >>"
except Exception:

    class SourceModule(object):

        def __init__(self, c):
            pass

        def get_function(self, name):
            pass
    cuda_can_help = False


class ProcessingStrategy(object):

    def __init__(self, strategy, loader, cache):
        self.initialize_slots(strategy, loader, cache)

    @property
    @memoize
    def decimalhour(self):
        int_to_dt = lambda t: datetime.utcfromtimestamp(t)
        int_to_decimalhour = (lambda time: int_to_dt(time).hour +
                      int_to_dt(time).minute/60.0 +
                      int_to_dt(time).second/3600.0)
        result = pmap(int_to_decimalhour, self.times)
        return np.array(result).reshape(self.times.shape)

    def create_1px_dimensions(self, cache):
        nc.getdim(cache, 'xc_k', 1)
        nc.getdim(cache, 'yc_k', 1)
        nc.getdim(cache, 'time', 1)

    def calculate_slots(self, images_per_hour):
        return np.round(self.decimalhour * images_per_hour).astype(int)

    def initialize_slots(self, strategy, loader, cache):
        self.create_1px_dimensions(cache)
        time = loader.time
        shape = list(time.shape)
        shape.append(1)
        self.times = time.reshape(tuple(shape))
        self.slots = cache.getvar('slots', 'i1', ('time', 'yc_k', 'xc_k'))
        self.slots[:] = self.calculate_slots(strategy.IMAGE_PER_HOUR)
        nc.sync(cache)



ma = np.ma
pi = str(np.float32(np.pi))
deg2rad_ratio = str(np.float32(np.pi / 180))
rad2deg_ratio = str(np.float32(180 / np.pi))


def gpu_exec(func, *matrixs):
    adapt = lambda m: m if isinstance(m, np.ndarray) else np.matrix(m)
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
    result = np.empty_like(matrixs[0])
    cuda.memcpy_dtoh(result, matrixs_gpu[0])
    for m in matrixs_gpu:
        m.free()
    # TODO: Try to change the api to return multiple results and with unfixed
    # shapes.
    return result


def spawn(f):
    def fun(pipe,x):
        pipe.send(f(x))
        pipe.close()
    return fun


def pmap(f,X):
    pipe=[Pipe() for x in X]
    proc=[Process(target=spawn(f),args=(c,x)) for x,(p,c) in izip(X,pipe)]
    [p.start() for p in proc]
    [p.join() for p in proc]
    return [p.recv() for (p,c) in pipe]
