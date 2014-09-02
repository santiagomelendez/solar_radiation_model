from libs.file import netcdf as nc
import os

def clone(files, filename, variables=['lat', 'lon', 'data']):
        obj_clone = nc.open(filename)[0]
        if not obj_clone.is_new:
            os.remove(filename)
        obj_clone.getdim('time')
        for d in files.roots[0].dimensions:
            if d != 'time':
                dim = files.roots[0].getdim(d)
                obj_clone.getdim(str(d), len(dim))
        # The first file set the variables origin
        variables = [str(v) for v in files.roots[0].variables.keys()
                     if v in variables]
        variables.sort()
        for v in variables:
            var_ref = files.roots[0].getvar(v)
	    if 'time' in var_ref.dimensions:            
	    	var = nc.getvar(files, v)
	    else:
		var = var_ref[:]
     	    var_clone = obj_clone.clonevar(var_ref, v)
            var_clone[:] = var[:]
	time = obj_clone.getvar('time', dimensions = ('time',))
	time[:] = nc.getvar(files, 'time')[:]
	nc.sync(obj_clone)
	return obj_clone, obj_clone.is_new

