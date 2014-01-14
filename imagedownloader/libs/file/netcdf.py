from netCDF4 import Dataset, numpy
import numpy as np
import os

def open(filename):
	is_new = not os.path.exists(filename)
	try:
		root = (Dataset(filename, mode='w',format='NETCDF4') if is_new else Dataset(filename,mode='a'))
	except Exception:
		root = Dataset(filename, mode='r')
	return root, is_new

def getdim(root,name,size=None):
	if not name in root.dimensions:
		d = root.createDimension(name, size)
	else:
		d = root.dimensions[name]
	return d

def getvar(root, name, vtype='f4', dimensions=(), digits=0):
	try:
		v = root.variables[name]
	except KeyError:
		if digits > 0:
			v = root.createVariable(name, vtype, dimensions, zlib=True, least_significant_digit=digits)
		else:
			v = root.createVariable(name, vtype, dimensions, zlib=True)
	return v

def sync(root):
	return root.sync()

def close(root):
	return root.close()

dtypes = {}
dtypes[numpy.dtype('float32')] = 'f4'
dtypes[numpy.dtype('int32')] = 'i4'

def clonevar(root, var, new_varname):
	var = getvar(root, var) if var.__class__ is str else var
	dims = [ str(d) for d in var.dimensions ]
	try:
		digit = var.least_significant_digit
	except AttributeError:
		digit = 0
	var_clone = getvar(root, new_varname, dtypes[var.dtype] , dims , digit )
	var_clone[:] = np.zeros(var_clone.shape)
	return var_clone

def clonefile(root, filename, avoided):
	if avoided is None: avoided = []
	variables = [ str(v) for v in root.variables.keys() if not v in avoided ]
	root_clone, is_new = open(filename)
	for d in root.dimensions:
		dim = getdim(root, d)
		getdim(root_clone, d, len(dim))
	for v in variables:
		var = getvar(root, v)
		var_clone = clonevar(root_clone, var, v)
		var_clone[:] = var[:]
	return root_clone, is_new

