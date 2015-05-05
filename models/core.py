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
        self.initialize_slots(loader, cache)
        self.create_variables(loader, cache)

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

    def initialize_slots(self, loader, cache):
        self.create_1px_dimensions(cache)
        time = loader.time
        shape = list(time.shape)
        shape.append(1)
        self.times = time.reshape(tuple(shape))
        self.slots = cache.getvar('slots', 'i1', ('time', 'yc_k', 'xc_k'))
        self.slots[:] = self.calculate_slots(self.algorithm.IMAGE_PER_HOUR)
        nc.sync(cache)

    def create_variables(self, loader, cache):
        create = lambda name, source: cache.getvar(name, source=source)
        self.declination = create('declination', self.slots)
        self.solarangle = cache.getvar('solarangle', 'f4',
                                       source=loader.ref_data)
        nc.sync(cache)
        self.solarelevation = create('solarelevation', self.solarangle)
        self.excentricity = create('excentricity', self.slots)
        self.gc = create('gc', self.solarangle)
        self.atmosphericalbedo = create('atmosphericalbedo', self.solarangle)
        self.t_sat = create('t_sat', loader.ref_lon)
        self.t_earth = create('t_earth', self.solarangle)
        self.cloudalbedo = create('cloudalbedo', self.solarangle)
        nc.sync(cache)
        if not os.path.exists('results'):
            os.makedirs('results')
        results_dir = lambda filename: filename.replace('temporal_cache', 'results')
        outputs = list(map(results_dir, cache.files))
        map(self.create_output_file, outputs)
        self.output, _ = nc.open(outputs)
        self.globalradiation = nc.getvar(self.output, 'globalradiation',
                                         source=self.solarangle)
        nc.sync(self.output)

    def create_output_file(self, filename):
        with nc.loader(filename) as output:
            nc.getdim(output, 'time', 1)


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
