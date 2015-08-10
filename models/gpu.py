import numpy as np
from netcdf import netcdf as nc
import stats
from helpers import show
from models.core import gpuarray, cuda, SourceModule
from cpu import CPUStrategy, GREENWICH_LON
import itertools
import math
import os
import signal
import subprocess

with open('models/kernel.cu') as f:
    mod_sourcecode = SourceModule(f.read())


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
    max_dims = getblockmaxdims(blocks[0], blocks[1], threads)
    blocks[0] = blocks[0] / max_dims[0]
    blocks[1] = blocks[1] / max_dims[1]
    show('-> grid dims: %s, block dims: %s, threads per block: %s\n' % (str(blocks), str([max_dims[0], max_dims[1], threads]), str(max_dims[0]*max_dims[1]*threads)))
    func(*matrixs_gpu, grid=tuple(blocks), block=tuple([max_dims[0], max_dims[1], threads]))
    list(map(lambda (m, m_gpu): cuda.memcpy_dtoh(m, m_gpu), transferences[:results]))
    for i in range(results):
        matrixs[i][:] = matrixs_ram[i]
        matrixs_gpu[i].free()
    return matrixs_ram[:results]


def getblockmaxdims(dimx, dimy, dimz):
    max_threads_per_block = 1024
    squares = cartesianproduct(list(divisors(dimx)), list(divisors(dimy)))
    validdims = filter(lambda x: x[0]*x[1]*dimz <= max_threads_per_block, squares)
    return max(validdims, key=lambda x: x[0]*x[1])


def divisors(n):
    large_divisors = []
    for i in xrange(1, int(math.sqrt(n) + 1)):
        if n % i is 0:
            yield i
            if i is not n / i:
                large_divisors.insert(0, n / i)
    for divisor in large_divisors:
        yield divisor


def cartesianproduct(x, y):
    return np.dstack(np.meshgrid(x, y)).reshape(-1, 2)


class GPUStrategy(CPUStrategy):

    def update_temporalcache(self, loader, cache):
        smi_proccess = subprocess.Popen("LD_LIBRARY_PATH=/usr/lib nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.total,memory.free,memory.used,temperature.gpu,timestamp --format=csv -l 2 -f smi-results.csv", shell=True, preexec_fn=os.setsid)
        const = lambda c: np.array(c).reshape(1, 1, 1)
        inputs = [loader.lat[0],
                  loader.lon[0],
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
            name = outputs[mm[0]].name if hasattr(outputs[mm[0]], 'name') else mm[0]
            print name, ': ', mm[1]
        print "----"
        nc.sync(cache)
        os.killpg(smi_proccess.pid, signal.SIGTERM)
        # super(GPUStrategy, self).update_temporalcache(loader, cache)


strategy = GPUStrategy
