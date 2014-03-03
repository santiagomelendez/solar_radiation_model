import sys
sys.path.append(".")
from django.db import models
from factopy.models import ComplexProcess, MaterialStatus, Importer, Adapt, Stream
#from plumbing import Stream
from requester.models import File
from datetime import datetime
from libs.file import netcdf as nc
import glob
import calendar
from libs import matrix
import pytz


class SyncImporter(Importer):
	class Meta(object):
			app_label = 'plumbing'

	def get_root_path_files(self):
		return glob.glob(self.stream.root_path + "*/*.nc")

	def update(self):
		materials_tmp = []
		for f in self.get_root_path_files():
			f, n = File.objects.get_or_create(localname=unicode(f))
			if n: f.save()
			ms, n = MaterialStatus.objects.get_or_create(material=f,stream=self.stream)
			if n: ms.save()
			materials_tmp.append([ms, True])
		return materials_tmp


class Program(ComplexProcess):
	class Meta(object):
			app_label = 'plumbing'
	stream = models.ForeignKey(Stream)

	def downloaded_files(self):
		return [ f for request in self.automatic_download.request_set.all() for f in request.order.file_set.filter(downloaded=True).order_by('localname') ]

	def source(self):
		files = self.downloaded_files()
		files.sort()
		return { 'all': files }

	def execute(self):
		self.do(self.stream)


class Compact(Adapt):
	class Meta(object):
			app_label = 'plumbing'
	extension = models.TextField()

	def do(self, stream):
		filename = "%spkg.%s.nc" % (self.stream.root_path,stream.tags.make_filename())
		f = self.do_file(filename,stream)
		if f:
			ms = MaterialStatus(material=f,stream=self.stream)
			ms.save()
		return self.stream

	def getdatetimenow(self):
		return datetime.utcnow().replace(tzinfo=pytz.UTC)

	def do_file(self, filename, stream):
		# create compact file and initialize basic settings
		if hasattr(stream.materials.all()[0].material,'completepath'):
			root, is_new = nc.open(filename)
			if is_new:
				sample = nc.open(stream.materials.all()[0].material.completepath())[0]
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
		else:
			return None

	def do_var(self, root, var_name, stream):
		material_statuses = sorted(stream.unprocessed(), key=lambda ms: ms.material.filename())
		shape = nc.getvar(root,'lat').shape
		for ms in material_statuses:
			# join the distributed content
			v_ch   = nc.getvar(root,var_name, 'f4', ('timing','northing','easting',), 4)
			v_ch_t = nc.getvar(root,var_name + '_time', 'f4', ('timing',))
			try:
				rootimg = nc.open(ms.material.completepath())[0]
				data = (nc.getvar(rootimg, 'data'))[:]
				# Force all the channels to the same shape
				if not (data.shape[1:3] == shape):
					print data.shape[1:3], shape
					data = matrix.adapt(data, shape)
				if v_ch.shape[1] == data.shape[1] and v_ch.shape[2] == data.shape[2]:
					index = v_ch.shape[0]
					v_ch[index,:] = data
					v_ch_t[index] = calendar.timegm(ms.material.datetime().utctimetuple())
				nc.close(rootimg)
				nc.sync(root)
			except RuntimeError:
				print ms.material.completepath()