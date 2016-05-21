from models import JobDescription
from models.cpu import CPUStrategy
from mpi4py import MPI
import numpy as np
import sys

job = JobDescription(data='data/goes13.2015.048.143733.BAND_01.nc',
                     product='prueba')

static = job.config['static_file']
loader = job.config['data']
print loader.filenames
time = loader.time
cpu = CPUStrategy(time)
data = cpu.getcalibrateddata(loader)
lat = static.lat
lon = static.lon
dem = static.dem
linke = static.linke
gr = np.empty(data.shape, dtype=np.float32)
ci = np.empty(data.shape, dtype=np.float32)
data_mix = np.empty(data.shape, dtype=np.float32)


comm = MPI.COMM_WORLD


def scatter_reshape(shape, size=comm.size, rank=comm.rank):
    lst_shape = list(shape)
    mod = shape[-1] % size
    tile = shape[-1] / size
    if rank < mod:
        tile = tile + 1
    lst_shape[-1] = tile
    return tuple(lst_shape)

p_data = np.empty(scatter_reshape(data.shape), dtype=np.float32)
p_lat = np.empty(scatter_reshape(lat.shape), dtype=np.float32)
p_lon = np.empty(scatter_reshape(lon.shape), dtype=np.float32)
p_dem = np.empty(scatter_reshape(dem.shape), dtype=np.float32)
p_linke = np.empty(scatter_reshape(linke.shape), dtype=np.float32)
p_gr = np.empty(scatter_reshape(gr.shape), dtype=np.float32)
p_ci = np.empty(scatter_reshape(ci.shape), dtype=np.float32)


comm.Scatterv([data, MPI.FLOAT], [p_data, MPI.FLOAT])
comm.Barrier()
comm.Scatterv([lat, MPI.FLOAT], [p_lat, MPI.FLOAT])
comm.Barrier()
comm.Scatterv([lon, MPI.FLOAT], [p_lon, MPI.FLOAT])
comm.Barrier()
comm.Scatterv([dem, MPI.FLOAT], [p_dem, MPI.FLOAT])
comm.Barrier()
comm.Scatterv([linke, MPI.FLOAT], [p_linke, MPI.FLOAT])
comm.Barrier()
comm.Scatterv([gr, MPI.FLOAT], [p_gr, MPI.FLOAT])
comm.Barrier()
comm.Scatterv([ci, MPI.FLOAT], [p_ci, MPI.FLOAT])
comm.Barrier()

cpu = CPUStrategy(time)
p_ci, p_gr = cpu.estimate_globalradiation(p_lat, p_lon, p_dem,
                                          p_linke, p_data)
p_ci = np.array(p_ci, dtype=np.float32)
comm.Barrier()
p_gr = np.array(p_gr, dtype=np.float32)
p_data[3, 40, :] = 60000
comm.Barrier()
comm.Gatherv([p_ci, MPI.FLOAT], [ci, MPI.FLOAT], root=0)
comm.Gatherv([p_gr, MPI.FLOAT], [gr, MPI.FLOAT], root=0)
comm.Gatherv([p_data, MPI.FLOAT], [data_mix, MPI.FLOAT], root=0)

if comm.rank != 0:
    sys.exit(0)

MPI.Finalize()


print 'min_gr: ', gr.min()
print 'max_gr: ', gr.max()
print 'min_ci: ', ci.min()
print 'max_ci: ', ci.max()
print 'Process finish.'
output = job.config['product']
output.ref_globalradiation[:] = gr
output.ref_cloudindex[:] = ci
output.ref_data_mix[:] = data_mix
sys.exit(0)
