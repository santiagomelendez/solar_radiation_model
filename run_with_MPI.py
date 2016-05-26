from models import JobDescription
from models.cpu import CPUStrategy
from mpi4py import MPI
import numpy as np
import sys

comm = MPI.COMM_WORLD


def scatter_reshape(shape, size=comm.size, rank=comm.rank, dim=-2):
    lst_shape = list(shape)
    mod = shape[dim] % size
    tile = shape[dim] / size
    if rank < mod:
        tile = tile + 1
    lst_shape[dim] = tile
    return tuple(lst_shape)

shape = np.empty(3, dtype=np.int)
notime_shape = np.empty(3, dtype=np.int)
linke_shape = np.empty(4, dtype=np.int)

if comm.rank == 0:
    job = JobDescription(data='data/goes13.2015.048.143733.BAND_01.nc',
                         product='mpi_prueba')
    time = job.config['data'].time
    cpu = CPUStrategy(time)
    data = cpu.getcalibrateddata(job.config['data'])
    lat = job.config['static_file'].lat
    lon = job.config['static_file'].lon
    dem = job.config['static_file'].dem
    gr = np.empty(data.shape, dtype=np.float64)
    ci = np.empty(data.shape, dtype=np.float64)
    linke = job.config['static_file'].linke
    shape = np.array(data.shape, dtype=np.int)
    linke_shape = np.array(linke.shape, dtype=np.int)
    notime_shape = np.array(lat.shape, dtype=np.int)
comm.Bcast([shape, MPI.INT], root=0)
if not comm.rank == 0:
    time = np.empty((shape[0], 1), np.int32)
comm.Bcast([linke_shape, MPI.INT], root=0)
comm.Bcast([notime_shape, MPI.INT], root=0)
comm.Bcast([time, MPI.INT], root=0)
comm.Barrier()
p_lat = np.empty(scatter_reshape(notime_shape), dtype=np.float64)
p_lon = np.empty(scatter_reshape(notime_shape), dtype=np.float64)
p_dem = np.empty(scatter_reshape(notime_shape), dtype=np.float64)
p_data = np.empty(scatter_reshape(shape), dtype=np.float64)
p_linke = np.empty(scatter_reshape(linke_shape), dtype=np.float64)
p_gr = np.empty(scatter_reshape(shape), dtype=np.float64)
p_ci = np.empty(scatter_reshape(shape), dtype=np.float64)

if comm.rank == 0:
    p_lat = lat[:, 0:p_lat.shape[-2], :]
    p_lon = lon[:, 0:p_lon.shape[-2], :]
    p_dem = dem[:, 0:p_dem.shape[-2], :]
    p_data = data[:, 0:p_data.shape[-2], :]
    p_linke = linke[:, :, 0:p_linke.shape[-2], :]
    pos = p_lat.shape[-2]
    for x in xrange(1, comm.size):
        rows = scatter_reshape(notime_shape, rank=x)[-2]
        count = np.cumproduct(scatter_reshape(notime_shape, rank=x))[-1]
        comm.Send([np.array(lat[:, pos: pos + rows, :], dtype=np.float64),
                   MPI.DOUBLE], dest=x, tag=1)
        comm.Send([np.array(lon[:, pos: pos + rows, :], dtype=np.float64),
                   MPI.DOUBLE], dest=x, tag=2)
        comm.Send([np.array(dem[:, pos: pos + rows, :], dtype=np.float64),
                   MPI.DOUBLE], dest=x, tag=3)
        comm.Send([np.array(linke[:, :, pos: pos + rows, :], dtype=np.float64),
                   MPI.DOUBLE], dest=x, tag=4)
        comm.Send([np.array(data[:, pos: pos + rows, :], dtype=np.float64),
                   MPI.DOUBLE], dest=x, tag=5)
        pos = pos + rows
else:
    cpu = CPUStrategy(time)
    comm.Recv([p_lat, MPI.DOUBLE], source=0, tag=1)
    comm.Recv([p_lon, MPI.DOUBLE], source=0, tag=2)
    comm.Recv([p_dem, MPI.DOUBLE], source=0, tag=3)
    comm.Recv([p_linke, MPI.DOUBLE], source=0, tag=4)
    comm.Recv([p_data, MPI.DOUBLE], source=0, tag=5)
p_ci, p_gr = cpu.estimate_globalradiation(p_lat, p_lon, p_dem,
                                          p_linke, p_data)
p_ci = np.array(p_ci, dtype=np.float64)
p_gr = np.array(p_gr, dtype=np.float64)
if comm.rank == 0:
    gr[:, 0:p_gr.shape[-2], :] = p_gr
    ci[:, 0:p_ci.shape[-2], :] = p_ci
    pos = p_gr.shape[-2]
    for x in xrange(1, comm.size):
        r_gr = np.empty(scatter_reshape(shape, rank=x), dtype=np.float64)
        r_ci = np.empty(scatter_reshape(shape, rank=x), dtype=np.float64)
        comm.Recv([r_gr, MPI.DOUBLE], source=x, tag=1)
        comm.Recv([r_ci, MPI.DOUBLE], source=x, tag=2)
        rows = r_gr.shape[-2]
        gr[:, pos:pos + rows, :] = r_gr
        ci[:, pos:pos + rows, :] = r_ci
        pos = pos + rows
    output = job.config['product']
    output.ref_cloudindex[:] = ci[-1, :, :]
    output.ref_globalradiation[:] = gr[-1, :, :]
else:
    comm.Send([p_gr, MPI.DOUBLE], dest=0, tag=1)
    comm.Send([p_ci, MPI.DOUBLE], dest=0, tag=2)
    sys.exit(0)
MPI.Finalize()
print 'Process finish.'
