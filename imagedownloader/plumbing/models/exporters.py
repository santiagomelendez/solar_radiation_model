from django.db import models
from core import Process, Stream, FileStatus, File
from libs.file import netcdf as nc
import os
import calendar
import collections
from libs import matrix


class Compact(Process):
	class Meta:
        	app_label = 'plumbing'
	extension = models.TextField()
	resultant_stream = models.ForeignKey(Stream, null=True, default=None)

	def do(self, stream):
		filename = "%spkg.%s.nc" % (self.resultant_stream.root_path,stream.tags.make_filename())
		file = self.do_file(filename,stream)
		fs = FileStatus(file=file,stream=self.resultant_stream)
		fs.save()
		return self.resultant_stream

	def getdatetimenow(self):
		return datetime.utcnow().replace(tzinfo=pytz.UTC)

	def do_file(self, filename, stream):
		# create compact file and initialize basic settings
		begin_time = self.getdatetimenow()
		root, is_new = nc.open(filename)
		if is_new:
			sample, n = nc.open(stream.files.all()[0].file.completepath())
			shape = sample.variables['data'].shape
			nc.getdim(root,'northing', shape[1])
			nc.getdim(root,'easting', shape[2])
			nc.getdim(root,'timing')
			v_lat = nc.getvar(root,'lat', 'f4', ('northing','easting',), 4)
			v_lon = nc.getvar(root,'lon', 'f4', ('northing','easting',), 4)
			v_lon[:] = nc.getvar(sample, 'lon')[:]
			v_lat[:] = nc.getvar(sample, 'lat')[:]
			nc.close(sample)
			nc.sync(root)
		self.do_var(root, 'data', stream)
		# save the content inside the compact file
		if not root is None: nc.close(root)
		f = File(localname=filename)
		f.save()
		return f

	def do_var(self, root, var_name, stream):
		count = 0
		file_statuses = stream.sorted_files()
		shape = nc.getvar(root,'lat').shape
		for fs in file_statuses:
			# join the distributed content
			ch = fs.file.channel()
			v_ch   = nc.getvar(root,var_name, 'f4', ('timing','northing','easting',), 4)
			v_ch_t = nc.getvar(root,var_name + '_time', 'f4', ('timing',))
			try:
				rootimg, n = nc.open(fs.file.completepath())
				data = (nc.getvar(rootimg, 'data'))[:]
				# Force all the channels to the same shape
				if not (data.shape[1:3] == shape):
					print data.shape[1:3], shape
					data = matrix.adapt(data, shape)
				if v_ch.shape[1] == data.shape[1] and v_ch.shape[2] == data.shape[2]:
					index = v_ch.shape[0]
					v_ch[index,:] = data
					v_ch_t[index] = calendar.timegm(fs.file.datetime().utctimetuple())
				nc.close(rootimg)
				nc.sync(root)
			except RuntimeError, e:
				print fs.file.completepath()
