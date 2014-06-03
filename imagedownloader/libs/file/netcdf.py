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

    def getvar(self, name, vtype='f4', dimensions=(), digits=0,
               fill_value=None):
        if name not in self.variables.keys():
            self.variables[name] = self.variable_wrapper(
                name,
                self.obtain_variable(
                    name, vtype, dimensions, digits, fill_value))
        return self.variables[name]


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

    def obtain_variable(self, name, vtype='f4', dimensions=(), digits=0,
                        fill_value=None):
        root = self.roots[0]
        return (root.variables[name] if name in root.variables.keys()
                else self.create_variable(name, vtype, dimensions,
                                          digits, fill_value))

    def create_variable(self, name, vtype='f4', dimensions=(), digits=0,
                        fill_value=None):
        root = self.roots[0]
        if digits > 0:
            extras = {'least_significant_digit': digits}
        return [root.createVariable(name, vtype, dimensions,
                                    zlib=True,
                                    fill_value=fill_value, *extras)]

    @property
    def dimensions(self):
        return self.roots[0].dimensions

    def getdim(self, name, size=None):
        if name not in self.dimensions:
            d = self.roots[0].createDimension(name, size)
        else:
            d = self.dimensions[name]
        return d

    def clonevar(self, varname, new_varname, extra_dimensions=[]):
        var = self.getvar(varname) if varname.__class__ is str else varname
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
        self.roots[0].sync()

    def close(self):
        self.roots[0].close()


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

    def obtain_variable(self, name, vtype='f4', dimensions=(), digits=0,
                        fill_value=None):
        return [f.getvar(name, vtype, dimensions, digits, fill_value)
                for f in self.roots]

    def getdim(self, name, size=None):
        return [r.getdim(name, size) for r in self.roots]

    def getvar(self, name, vtype='f4', dimensions=(), digits=0,
               fill_value=None):
        vars = DistributedNCVariable(
            name,
            [r.getvar(name, vtype, dimensions, digits)[:]
             for r in self.roots])
        return vars

    def sync(self):
        [r.sync() for r in self.roots]

    def close(self):
        [r.close() for r in self.roots]

    def clonevar(self, var, new_varname):
        return DistributedNCVariable(
            new_varname,
            [r.clonevar(var, new_varname) for r in self.roots])

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
            var = self.getvar(v).pack()
            var_clone = obj_clone.clonevar(var_ref, v, ['time'])
            try:
                var_clone[:] = var[:]
            except Exception, e:
                print e
        obj_clone.sync()
        return obj_clone, obj_clone.is_new


class NCVariable(object):

    def __init__(self, name, variables):
        self.name = name
        self.variables = variables

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
        print self.__class__, self.name, name

    def sync(self):
        for v in self.variables:
            v.group().sync()


class SingleNCVariable(NCVariable):

    def pack(self):
        # var = var[:]
        # if 'S' in str(var.dtype):
        #    var = np.vstack([var.tostring()])
        return self.variables[0]

    def __setitem__(self, indexes, changes):
        return self.pack().__setitem__(indexes, changes)


class DistributedNCVariable(NCVariable):

    def pack(self):
        s = self.variables[0].shape
        if len(s) > 0:
            if 'S' in str(self.variables[0].dtype):
                res = [v.tostring() for v in self.variables]
            elif s[0] > 1:
                res = [self.variables]
            else:
                res = self.variables
            return np.vstack(res)

    def __setitem__(self, indexes, change):
        try:
            l = np.vsplit(change, change.shape[0])
        except Exception, e:
            print e
            # TODO: vsplit slice object should avoid change.shape[0]
        ifnone = lambda a, b: b if a is None else a
        item = indexes
        idx = indexes
        if isinstance(indexes, tuple):
            item = indexes[0]
            idx = tuple([slice(None)] + list(indexes[1:]))
        filtered = list(range(ifnone(item.start, 0),
                              ifnone(item.stop, change.shape[0]),
                              ifnone(item.step, 1)))
        for i in filtered:
            v = self.variables[i]
            v[idx] = l[i]
        self.sync()


def open(pattern):
    obj = NCObject.open(pattern)
    return obj, obj.is_new


def getdim(obj, name, size=None):
    return obj.getdim(name, size)


def getvar(obj, name, vtype='f4', dimensions=(), digits=0, fill_value=None):
    return obj.getvar(name, vtype, dimensions, digits, fill_value)


def sync(obj):
    obj.sync()


def close(obj):
    obj.close()


def clonevar(obj, var, new_varname):
    return obj.clonevar(var, new_varname)


def clone(obj, filename, avoided=[]):
    return obj.clone(filename, avoided)
