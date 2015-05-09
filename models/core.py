from datetime import datetime
import numpy as np
from netcdf import netcdf as nc
from multiprocessing import Process, Pipe
from itertools import izip
from cache import memoize
import multiprocessing as mp
import os


try:
    raise Exception('Force CPU')
    from pycuda.compiler import SourceModule
    import pycuda.gpuarray as gpuarray
    import pycuda.driver as cuda
    import pycuda.autoinit
    cuda_can_help = True
    print "<< using CUDA cores >>"
except Exception:
    cuda_can_help = False


class ProcessingStrategy(object):

    def __init__(self, algorithm, loader, cache):
        self.algorithm = algorithm
        self.algorithm.create_variables(loader, cache, self)

    @property
    @memoize
    def decimalhour(self):
        int_to_dt = lambda t: datetime.utcfromtimestamp(t)
        int_to_decimalhour = (lambda time: int_to_dt(time).hour +
                              int_to_dt(time).minute/60.0 +
                              int_to_dt(time).second/3600.0)
        result = pmap(int_to_decimalhour, self.times)
        return np.array(result).reshape(self.times.shape)

    def calculate_slots(self, images_per_hour):
        return np.round(self.decimalhour * images_per_hour).astype(int)


def spawn(f):
    def fun(pipe, x):
        pipe.send(f(x))
        pipe.close()
    return fun


def mp_map(f, X):
    pipe = [Pipe() for x in X]
    proc = [Process(target=spawn(f), args=(c, x))
            for x, (p, c) in izip(X, pipe)]
    [p.start() for p in proc]
    [p.join() for p in proc]
    return [p.recv() for (p, c) in pipe]


pmap = map if 'armv6l' in list(os.uname()) else mp_map
