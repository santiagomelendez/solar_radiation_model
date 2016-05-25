from models import JobDescription
from models.cpu import CPUStrategy
from mpi4py import MPI
import numpy as np
import sys

job = JobDescription(data='data/goes13.2015.048.143733.BAND_01.nc',
                     product='mpi_prueba')

static = job.config['static_file']
loader = job.config['data']
time = loader.time
cpu = CPUStrategy(time)
data = cpu.getcalibrateddata(loader)
print data.shape
lat = static.lat
lon = static.lon
dem = static.dem
linke = static.linke
print lat.shape
gr = np.empty(data.shape, dtype=np.float32)
ci = np.empty(data.shape, dtype=np.float32)

comm = MPI.COMM_WORLD


"""
def scatter_reshape(shape, size=comm.size, rank=comm.rank, dim=-2):
    lst_shape = list(shape)
    mod = shape[dim] % size
    tile = shape[dim] / size
    if rank < mod:
        tile = tile + 1
    lst_shape[dim] = tile
    return tuple(lst_shape)
"""


def get_splited_data(var, size=comm.size, rank=comm.rank):
    splited_shapes = np.array_split(var, size, axis=-2)
    count = tuple(map(lambda s_var: np.cumproduct(s_var.shape)[-1],
                      splited_shapes))
    splited_arrays = map(lambda s_var: np.empty(s_var.shape, dtype=np.float32),
                         splited_shapes)
    return splited_arrays[rank], count


p_lat, lat_count = get_splited_data(lat)
p_lon, lon_count = get_splited_data(lon)
p_dem, dem_count = get_splited_data(dem)
p_linke, linke_count = get_splited_data(linke)
p_data, data_count = get_splited_data(data)
p_gr, gr_count = get_splited_data(gr)
p_ci, ci_count = get_splited_data(ci)


comm.Scatterv([lat, lat_count, tuple(range(comm.size)), MPI.FLOAT],
              [p_lat, MPI.FLOAT])
comm.Scatterv([lon, lon_count, tuple(range(comm.size)), MPI.FLOAT],
              [p_lon, MPI.FLOAT])
comm.Scatterv([dem, dem_count, tuple(range(comm.size)), MPI.FLOAT],
              [p_dem, MPI.FLOAT])
comm.Scatterv([linke, linke_count, tuple(range(comm.size)), MPI.FLOAT],
              [p_linke, MPI.FLOAT])
comm.Scatterv([data, data_count, tuple(range(comm.size)), MPI.FLOAT],
              [p_data, MPI.FLOAT])
comm.Scatterv([gr, gr_count, tuple(range(comm.size)), MPI.FLOAT],
              [p_gr, MPI.FLOAT])
comm.Scatterv([ci, ci_count, tuple(range(comm.size)), MPI.FLOAT],
              [p_ci, MPI.FLOAT])
comm.Barrier()
"""
cpu = CPUStrategy(time)
p_ci, p_gr = cpu.estimate_globalradiation(p_lat, p_lon, p_dem,
                                          p_linke, p_data)
p_ci = np.array(p_ci, dtype=np.float32)
p_gr = np.array(p_gr, dtype=np.float32)
"""
p_gr = (comm.rank + 1) * np.ones_like(p_data)

comm.Barrier()

print "I am [%0d] : %s" % (comm.rank, p_gr.shape)
comm.Barrier()

if comm.rank != 0:
    comm.Send([p_gr, MPI.FLOAT], dest=0, tag=1)
    #comm.Send([p_ci, MPI.FLOAT], dest=0, tag=2)

if comm.rank == 0:
    gr[:, 0:p_gr.shape[1], :] = p_gr
    ci[:, 0:p_ci.shape[1], :] = p_ci
    for x in range(1, comm.size):
        r_gr = get_splited_data(data, rank=x)[0]
        #r_ci = get_splited_data(data, rank=x)[0]
        comm.Recv([r_gr, gr_count[x], MPI.FLOAT], source=x, tag=1)
        #comm.Recv([r_ci, MPI.FLOAT], source=x, tag=2)
        tile = '%s:%s' % (r_gr.shape[-2]*x, r_gr.shape[-2]*(x+1))
        print "globalradiation splited shape: ", tile
        gr[:, r_gr.shape[-2]*x:r_gr.shape[-2]*(x+1), :] = r_gr
        #ci[:, r_ci.shape[-2]*x:r_ci.shape[-2]*(x+1), :] = r_ci


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
sys.exit(0)
