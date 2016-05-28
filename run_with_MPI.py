from models.cpu import CPUStrategy
from models.temporal_serie import TemporalSerie
from models.cache import OutputCache
from datetime import datetime, timedelta
from mpi4py import MPI
import numpy as np
import netCDF4 as nc
import sys

comm = MPI.COMM_WORLD
dim = -2


def scatter_reshape(shape, size=comm.size, rank=comm.rank, dim=dim):
    lst_shape = list(shape)
    mod = shape[dim] % size
    tile = shape[dim] / size
    if rank < mod:
        tile = tile + 1
    lst_shape[dim] = tile
    return tuple(lst_shape)

start = MPI.Wtime()
shape = np.empty(3, dtype=np.int)
notime_shape = np.empty(3, dtype=np.int)
linke_shape = np.empty(4, dtype=np.int)

if comm.rank == 0:
    start_p = datetime.now()
    image = 'data/goes13.2015.048.143733.BAND_01.nc'
    product = 'mpi_results'
    data = TemporalSerie(image).get()
    root_data = nc.MFDataset(data, aggdim='time')
    rdv = root_data.variables
    print '-'*20, 'Getting data to process', '-'*20
    data = rdv['data'][:]
    print '-'*20, 'Getting time to process', '-'*20
    time = rdv['time'][:].reshape((len(data), 1))
    cpu = CPUStrategy(time)
    print '-'*20, 'Calibrating data', '-'*20
    data = cpu.getcalibrateddata(data, rdv['counts_shift'][:],
                                 rdv['space_measurement'][:],
                                 rdv['prelaunch_0'][:],
                                 rdv['postlaunch'][:])
    root_data.close()
    root_static = nc.Dataset('static.nc', 'r')
    rdv = root_static.variables
    reshape_var = lambda var: var.reshape(tuple([1] + list(var.shape)))
    print '-'*20, 'Getting lat to process', '-'*20
    lat = reshape_var(rdv['lat'][:])
    print '-'*20, 'Getting lon to process', '-'*20
    lon = reshape_var(rdv['lon'][:])
    print '-'*20, 'Getting dem to process', '-'*20
    dem = reshape_var(rdv['dem'][:])
    print '-'*20, 'Getting linke to process', '-'*20
    linke = reshape_var(rdv['linke'][:])
    root_static.close()
    gr = np.empty(data.shape, dtype=np.float64)
    ci = np.empty(data.shape, dtype=np.float64)
    shape = np.array(data.shape, dtype=np.int)
    linke_shape = np.array(linke.shape, dtype=np.int)
    notime_shape = np.array(lat.shape, dtype=np.int)
    end_p = datetime.now()
    print 'time to load the data model: ', (end_p - start_p).total_seconds()
    print '-'*10, 'Broadcast the shape of matrix', '-'*10
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
    p_lat = lat[:, 0:p_lat.shape[dim], :]
    p_lon = lon[:, 0:p_lon.shape[dim], :]
    p_dem = dem[:, 0:p_dem.shape[dim], :]
    p_data = data[:, 0:p_data.shape[dim], :]
    p_linke = linke[:, :, 0:p_linke.shape[dim], :]
    pos = p_lat.shape[dim]
    for x in xrange(1, comm.size):
        rows = scatter_reshape(notime_shape, rank=x)[dim]
        count = np.cumproduct(scatter_reshape(notime_shape, rank=x))[-1]
        print'node 0: Sending the variables to node %x' % (x)
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
    print 'node %d: Reciving variables from node 0' % (comm.rank)
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
    gr[:, 0:p_gr.shape[dim], :] = p_gr
    ci[:, 0:p_ci.shape[dim], :] = p_ci
    pos = p_gr.shape[dim]
    for x in xrange(1, comm.size):
        r_gr = np.empty(scatter_reshape(shape, rank=x), dtype=np.float64)
        r_ci = np.empty(scatter_reshape(shape, rank=x), dtype=np.float64)
        print'node 0: Reciving th calculate variables by node: %i' % (x)
        comm.Recv([r_gr, MPI.DOUBLE], source=x, tag=1)
        comm.Recv([r_ci, MPI.DOUBLE], source=x, tag=2)
        rows = r_gr.shape[dim]
        gr[:, pos:pos + rows, :] = r_gr
        ci[:, pos:pos + rows, :] = r_ci
        pos = pos + rows
    output = OutputCache(product, {}, [image])
    output.ref_cloudindex[:] = ci[-1, :, :]
    output.ref_globalradiation[:] = gr[-1, :, :]
else:
    comm.Send([p_gr, MPI.DOUBLE], dest=0, tag=1)
    comm.Send([p_ci, MPI.DOUBLE], dest=0, tag=2)
    sys.exit(0)
end = MPI.Wtime()
MPI.Finalize()
print 'total time: ', end - start
print 'Process finish.'
