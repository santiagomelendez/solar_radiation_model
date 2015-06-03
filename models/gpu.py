import numpy as np
from netcdf import netcdf as nc
import stats
from helpers import show
from models.core import gpuarray, cuda, SourceModule
from cpu import CPUStrategy, GREENWICH_LON
import itertools

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
        # super(GPUStrategy, self).update_temporalcache(loader, cache)


strategy = GPUStrategy
