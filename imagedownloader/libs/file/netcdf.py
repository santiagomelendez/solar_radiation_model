from netCDF4 import Dataset, numpy
import numpy as np
import os
from glob import glob


dtypes = {}
dtypes[numpy.dtype('float32')] = 'f4'
dtypes[numpy.dtype('int32')] = 'i4'
dtypes[numpy.dtype('int8')] = 'i1'
dtypes[numpy.dtype('S1')] = 'S1'


class NCObject(object):

    @classmethod
    def open(self, pattern):
        if pattern.__class__ is str and '*' not in pattern:
            o = File(pattern=pattern)
        else:
            if pattern.__class__ is list:
                o = Package(files=pattern)
            else:
                o = Package(pattern=pattern)
        o.load()
        return o

    def __init__(self, pattern='', files=[]):
        super(NCObject, self).__init__()
        self.pattern = pattern
        if len(files):
            self.files = files
        else:
            self.files = glob(pattern)
        self.files.sort()
        self.is_new = not os.path.exists(pattern)


class File(NCObject):

    def load(self):
        filename = self.pattern if (len(self.files) == 0 and
                                    len(self.pattern)) else self.files[0]
        try:
            self.roots = [(Dataset(filename, mode='w', format='NETCDF4')
                          if self.is_new else Dataset(filename, mode='a',
                                                      format='NETCDF4'))]
        except Exception:
            self.roots = [Dataset(filename, mode='r', format='NETCDF4')]

    @property
    def root(self):
        return self.roots[0]

    @property
    def dimensions(self):
        return self.root.dimensions

    @property
    def variables(self):
        return self.root.variables

    def getdim(self, name, size=None):
        if name not in self.dimensions:
            d = self.root.createDimension(name, size)
        else:
            d = self.dimensions[name]
        return d

    def getvar(self, name, vtype='f4', dimensions=(), digits=0,
               fill_value=None):
        try:
            v = self.variables[name]
        except KeyError:
            if digits > 0:
                v = self.root.createVariable(name, vtype,
                                             dimensions, zlib=True,
                                             least_significant_digit=digits,
                                             fill_value=fill_value)
            else:
                v = self.root.createVariable(name, vtype, dimensions,
                                             zlib=True,
                                             fill_value=fill_value)
        return v

    def clonevar(self, varname, new_varname, extra_dimensions=[]):
        var = self.variables[varname] if varname.__class__ is str else varname
        dims = list(var.dimensions)
        for ed in extra_dimensions:
            if ed not in dims:
                dims.insert(0, ed)
        try:
            digit = var.least_significant_digit
        except AttributeError:
            digit = 0
        var_clone = self.getvar(new_varname, dtypes[var.dtype], dims, digit)
        var_clone[:] = np.zeros(var_clone.shape)
        return var_clone

    def clone(self, filename, avoided):
        if avoided is None:
            avoided = []
        variables = [str(v) for v in self.variables.keys()
                     if v not in avoided]
        obj_clone, is_new = NCObject.open(filename)
        for d in self.dimensions:
            dim = self.getdim(d)
            obj_clone.getdim(d, len(dim))
        for v in variables:
            var = self.getvar(v)
            var_clone = obj_clone.clonevar(var, v)
            var_clone[:] = var[:]
        return obj_clone, is_new

    def sync(self):
        self.root.close()

    def close(self):
        self.root.close()

    def pack(self, var):
        var = var[:]
        if 'S' in str(var.dtype):
            var = np.vstack([var.tostring()])
        return var


class Package(NCObject):

    def load(self):
        self.roots = [NCObject.open(filename) for filename in self.files]

    def getdim(self, name, size=None):
        return [r.getdim(name, size) for r in self.roots]

    def getvar(self, name, vtype='f4', dimensions=(), digits=0,
               fill_value=None):
        vars = [r.getvar(name, vtype, dimensions, digits)[:]
                for r in self.roots]
        return vars

    def sync(self):
        [r.sync() for r in self.roots]

    def close(self):
        [r.close() for r in self.roots]

    def clonevar(self, var, new_varname):
        return [r.clonevar(var, new_varname) for r in self.roots]

    def clone(self, filename, avoided=[]):
        obj_clone = NCObject.open(filename)
        if not obj_clone.is_new:
            os.remove(filename)
        obj_clone.getdim('time')
        for d in self.roots[0].dimensions:
            if d != 'time':
                dim = self.getdim(d)
                obj_clone.getdim(str(d), len(dim[0]))
        # The first file set the variables origin
        variables = [str(v) for v in self.roots[0].variables.keys()
                     if v not in avoided]
        variables.sort()
        for v in variables:
            var_ref = self.roots[0].getvar(v)
            var = getvar(self, v)
            var_clone = obj_clone.clonevar(var_ref, v, ['time'])
            var_clone[:] = var[:]
        return obj_clone, obj_clone.is_new

    def pack(self, var):
        s = var[0].shape
        if len(s) > 0:
            if 'S' in str(var[0].dtype):
                res = [v.tostring() for v in var]
            elif s[0] > 1:
                res = [var]
            else:
                res = var
            return np.vstack(res)


def open(pattern):
    obj = NCObject.open(pattern)
    return obj, obj.is_new


def getdim(obj, name, size=None):
    return obj.getdim(name, size)


def getvar(obj, name, vtype='f4', dimensions=(), digits=0, fill_value=None):
    return obj.pack(obj.getvar(name, vtype, dimensions, digits, fill_value))


def sync(obj):
    obj.sync()


def close(obj):
    obj.close()


def clonevar(obj, var, new_varname):
    return obj.clonevar(var, new_varname)


def clone(obj, filename, avoided=[]):
    return obj.clone(filename, avoided)
