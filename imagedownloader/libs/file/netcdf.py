from netCDF4 import Dataset, numpy
import numpy as np
import os
from glob import glob
from compiler.ast import flatten


dtypes = {}
dtypes[numpy.dtype('float32')] = 'f4'
dtypes[numpy.dtype('int32')] = 'i4'
dtypes[numpy.dtype('int8')] = 'i1'
dtypes[numpy.dtype('S1')] = 'S1'


class NCObject(object):

    @classmethod
    def open(self, files_or_pattern):
        files, pattern = self.distill(files_or_pattern)
        o = self.choice_type(files)
        o.pattern = pattern
        o.load()
        return o

    @classmethod
    def choice_type(self, files):
        return NCPackage(files) if len(files) > 1 else NCFile(files)

    @classmethod
    def distill(self, files_or_pattern):
        if files_or_pattern.__class__ is list:
            files = files_or_pattern
            pattern = ''
        else:
            files = glob(files_or_pattern)
            pattern = files_or_pattern
        if not len(files):
            files = [files_or_pattern]
            pattern = files_or_pattern
        return files, pattern

    def __init__(self, files):
        super(NCObject, self).__init__()
        self.files = files
        self.files.sort()
        self.variables = {}
        self._is_new = [not os.path.exists(f) for f in self.files]

    @property
    def dimensions(self):
        dicts = [r.dimensions for r in self.roots]
        keys = {k for d in dicts for k in d}
        return {k: flatten([d.get(k) for d in dicts])
                for k in keys}

    def has_dimension(self, name):
        return all([name in r.dimensions.keys() for r in self.roots])

    def getdim(self, name, size=None):
        return (self.obtain_dimension(name)
                if self.has_dimension(name)
                else self.create_dimension(name, size))

    def getvar(self, name, vtype='f4', dimensions=(), digits=0,
               fill_value=None):
        if name not in self.variables.keys():
            vars = self.obtain_variable(name, vtype, dimensions,
                                        digits, fill_value)
            self.variables[name] = self.variable_wrapper(name, vars)
        return self.variables[name]

    def sync(self):
        return [r.sync() for r in self.roots]

    def close(self):
        return [r.close() for r in self.roots]


class NCFile(NCObject):

    @property
    def is_new(self):
        return self._is_new[0]

    def load(self):
        filename = self.files[0]
        self.read_only = True
        try:
            self.roots = [(Dataset(filename, mode='w', format='NETCDF4')
                          if self.is_new else Dataset(filename, mode='a',
                                                      format='NETCDF4'))]
            self.read_only = False
        except Exception:
            self.roots = [Dataset(filename, mode='r', format='NETCDF4')]
        self.variable_wrapper = SingleNCVariable

    def obtain_dimension(self, name):
        return [r.dimensions[name] for r in self.roots]

    def create_dimension(self, name, size):
        return [r.createDimension(name, size) for r in self.roots]

    def obtain_variable(self, name, vtype='f4', dimensions=(), digits=0,
                        fill_value=None):
        root = self.roots[0]
        return (root.variables[name] if name in root.variables.keys()
                else self.create_variable(name, vtype, dimensions,
                                          digits, fill_value))

    def create_variable(self, name, vtype='f4', dimensions=(), digits=0,
                        fill_value=None):
        build = self.roots[0].createVariable
        options = {'zlib': True,
                   'fill_value': fill_value}
        if digits > 0:
            options['least_significant_digit'] = digits
        return [build(name, vtype, dimensions, *options)]

    # def clonevar(self, varname, new_varname, extra_dimensions=[]):
    #    var = self.getvar(varname) if varname.__class__ is str else varname
    #    dims = list(var.dimensions)
    #    for ed in extra_dimensions:
    #        if ed not in dims:
    #            dims.insert(0, ed)
    #    try:
    #        digit = var.least_significant_digit
    #    except AttributeError:
    #        digit = 0
    #    var_clone = self.getvar(new_varname, dtypes[var.dtype], dims, digit)
    #    var_clone[:] = np.zeros(var_clone.shape)
    #    return var_clone

    # def clone(self, filename, avoided):
    #    if avoided is None:
    #        avoided = []
    #    variables = [str(v) for v in self.variables.keys()
    #                 if v not in avoided]
    #    obj_clone, is_new = NCObject.open(filename)
    #    for d in self.dimensions:
    #        dim = self.getdim(d)
    #        obj_clone.getdim(d, len(dim))
    #    for v in variables:
    #        var = self.getvar(v)
    #        var_clone = obj_clone.clonevar(var, v)
    #        var_clone[:] = var[:]
    #    return obj_clone, is_new


class NCPackage(NCObject):

    @property
    def is_new(self):
        return all(self._is_new)

    def load(self):
        self.roots = [NCObject.open(filename) for filename in self.files]
        self.variable_wrapper = DistributedNCVariable

    @property
    def read_only(self):
        return all([r.read_only for r in self.roots])

    def obtain_dimension(self, name):
        return self.dimensions[name]

    def create_dimension(self, name, size):
        return [r.create_dimension(name, size) for r in self.roots]

    def obtain_variable(self, name, vtype='f4', dimensions=(), digits=0,
                        fill_value=None):
        return [r.getvar(name, vtype, dimensions, digits, fill_value)
                for r in self.roots]

    # def clonevar(self, var, new_varname):
    #    return DistributedNCVariable(
    #        new_varname,
    #        [r.clonevar(var, new_varname) for r in self.roots])

    # def clone(self, filename, avoided=[]):
    #    obj_clone = NCObject.open(filename)
    #    if not obj_clone.is_new:
    #        os.remove(filename)
    #    obj_clone.getdim('time')
    #    for d in self.roots[0].dimensions:
    #        if d != 'time':
    #            dim = self.getdim(d)
    #            obj_clone.getdim(str(d), len(dim[0]))
    #    # The first file set the variables origin
    #    variables = [str(v) for v in self.roots[0].variables.keys()
    #                 if v not in avoided]
    #    variables.sort()
    #    for v in variables:
    #        var_ref = self.roots[0].getvar(v)
    #        var = self.getvar(v).pack()
    #        var_clone = obj_clone.clonevar(var_ref, v, ['time'])
    #        try:
    #            var_clone[:] = var[:]
    #        except Exception, e:
    #            print e
    #    obj_clone.sync()
    #    return obj_clone, obj_clone.is_new


class NCVariable(object):

    def __init__(self, name, variables):
        self.name = name
        self.variables = (variables
                          if variables.__class__ is list else [variables])

    def __eq__(self, obj):
        return (self.pack() == obj).all()

    @property
    def shape(self):
        return self.pack().shape

    @property
    def dimensions(self):
        return self.pack().dimensions

    @property
    def least_significant_digit(self):
        v = self.pack()
        return (v.least_significant_digit
                if hasattr(v, 'least_significant_digit') else 0)

    @property
    def dtype(self):
        return self.pack().dtype

    def __getitem__(self, indexes):
        return self.pack().__getitem__(indexes)

    def __getattr__(self, name):
        print 'Unhandled [class: %s, instance: %s, attr: %s]' % (
            self.__class__, self.name, name)

    def sync(self):
        for v in self.variables:
            v.group().sync()


class SingleNCVariable(NCVariable):

    def group(self):
        return self.variables[0].group()

    def pack(self):
        vars = self.variables[0]
        if self.variables[0].shape[0] > 1:
            vars = np.vstack([self.variables])
        return vars

    def __setitem__(self, indexes, changes):
        return self.variables[0].__setitem__(indexes, changes)


class DistributedNCVariable(NCVariable):

    def pack(self):
        return np.vstack([v.pack() for v in self.variables])

    def __setitem__(self, indexes, change):
        pack = self.pack()
        pack.__setitem__(indexes, change)
        vars = np.vsplit(pack, pack.shape[0])
        for i in range(len(vars)):
            self.variables[i][:] = vars[i]
        self.sync()


def open(pattern):
    """
    Open a root descriptor to work with one or multiple NetCDF files.
    """
    obj = NCObject.open(pattern)
    return obj, obj.is_new


def getdim(obj, name, size=None):
    """
    Return the dimension list of a NCFile or NCPackage instance.
    """
    dim = obj.getdim(name, size)
    return dim


def getvar(obj, name, vtype='f4', dimensions=(), digits=0, fill_value=None):
    """
    Return the numpy matrix of a variable from aNCFile or NCPackage instance.
    """
    return obj.getvar(name, vtype, dimensions, digits, fill_value)


def sync(obj):
    """
    Force the root descriptor to synchronize writing the buffers to the disk.
    """
    obj.sync()


def close(obj):
    """
    Close the root descriptor and write the buffer to the disk.
    """
    obj.close()

"""
def clonevar(obj, var, new_varname):
    return obj.clonevar(var, new_varname)


def clone(obj, filename, avoided=[]):
    return obj.clone(filename, avoided)
"""
