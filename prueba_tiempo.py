from mpi4py import MPI
from netCDF4 import Dataset
import numpy as np
import sys


data_mix = np.zeros((30, 254, 400), dtype=np.float32)

comm = MPI.COMM_WORLD


def scatter_reshape(shape, size=comm.size, rank=comm.rank, dim=-2):
    lst_shape = list(shape)
    mod = shape[dim] % size
    tile = shape[dim] / size
    if rank < mod:
        tile = tile + 1
    lst_shape[dim] = tile
    return tuple(lst_shape)


p_data = np.empty(scatter_reshape(data_mix.shape), dtype=np.float32)

print"I am [%d] and I send to 0 this shape: %s" % (comm.rank, p_data.shape)

tiles = map(lambda x: scatter_reshape(data_mix.shape, rank=x),
            range(comm.size))

counts = tuple(map(lambda t: np.cumproduct(t)[-1], tiles))
comm.Scatterv([data_mix, counts, tuple(range(comm.size)), MPI.FLOAT],
              [p_data, MPI.FLOAT])

comm.Barrier()
p_data = (comm.rank + 1) * np.ones_like(p_data)
print counts
comm.Barrier()
comm.Gatherv([p_data, MPI.FLOAT], [data_mix, counts,
                                   tuple(range(comm.size)), MPI.FLOAT], root=0)

"""
if comm.rank != 0:
    comm.Send([p_data, MPI.FLOAT], dest=0, tag=1)

if comm.rank == 0:
    data_mix[:, 0:p_data.shape[1], :] = p_data
    for x in range(1, comm.size):
        r_data = np.empty(scatter_reshape(data_mix.shape, rank=x), np.float32)
        comm.Recv([r_data, MPI.FLOAT], dest=x, tag=1)
        r_data.shape
        tile = '%s:%s' % (r_data.shape[1]*x, r_data.shape[1]*(x+1))
        print tile
        data_mix[:, r_data.shape[1]*x:r_data.shape[1]*(x+1), :] = r_data
        r_data = None
"""

if comm.rank != 0:
    sys.exit(0)

MPI.Finalize()

print 'Process finish.'

root = Dataset('prueba/data_mix.nc', 'w')
root.createDimension('time', 30)
root.createDimension('yc', 254)
root.createDimension('xc', 400)
data = root.createVariable('data_mix', np.float32, dimensions=('time',
                                                               'yc', 'xc'))

print data.shape
data[:] = data_mix
root.sync()
root.close()

sys.exit(0)
