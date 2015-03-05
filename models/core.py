import numpy as np


try:
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


