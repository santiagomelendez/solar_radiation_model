import sys
sys.path.append(".")
from django.db import models
from core import File, ComplexProcess, Stream, Adapter
from datetime import datetime, timedelta
import re
from libs.file import netcdf as nc
import os
import threading
import glob


class Importer(Adapter):
	class Meta(object):
			app_label = 'plumbing'
	frequency = models.IntegerField(default=15*60) # It is expressed in seconds

	@classmethod
	def setup_unloaded(klass):
		importers = [ i for i in klass.objects.all() if not hasattr(i,thread) ]
		for i in importers:
			i.thread = threading.Timer(i.frequency, i.update).start()
		return len(importers)


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
			if n: fs.save()
			materials_tmp.append([ms, True])
		return materials_tmp


class Image(File):
	class Meta(object):
		app_label = 'plumbing'

	def channel(self):
		res = re.search('BAND_([0-9]*)\.', self.completepath())
		return str(res.groups(0)[0]) if res else None

	def satellite(self):
		res = self.filename().split(".")
		return str(res[0])

	def datetime(self):
		t_info = self.filename().split(".")
		year = int(t_info[1])
		days = int(t_info[2])
		time = t_info[3]
		date = datetime(year, 1, 1) + timedelta(days - 1)
		return date.replace(hour=int(time[0:2]), minute=int(time[2:4]), second=int(time[4:6]))

	def latlon(self):
		if self.channel() is None:
			return None, None
		root = nc.open(self.completepath())[0]
		lat = nc.getvar(root, "lat")[:]
		lon = nc.getvar(root, "lon")[:]
		nc.close(root)
		return lat, lon

	def completepath(self):
		return os.path.expanduser(os.path.normpath(self.localname))


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