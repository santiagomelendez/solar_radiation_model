from django.db import models
from requester.models import *
from process import *
import numpy as np
from libs.geometry import jaen as geo
from libs.file import netcdf as nc

class FilterSolarElevation(Process):
	class Meta:
        	app_label = 'plumbing'
	minimum = models.DecimalField(max_digits=4,decimal_places=2)
	hourly_longitude = models.DecimalField(max_digits=5, decimal_places=2, default=Decimal('0.00'))
	def do(self,stream):
		resultant_stream = stream.clone()
		for fs in stream.files.all():
			sub_lon = float(self.hourly_longitude)
			lat,lon = fs.file.latlon()
			if self.solar_elevation(fs.file, sub_lon, lat, lon) >= self.minimum:
				fs.clone_for(resultant_stream)
			fs.processed=True
			fs.save()
		return resultant_stream
	def solar_elevation(self, f, sub_lon, lat, lon):
		solarelevation_min = -90.0
		if not lon is None:
			solarelevation = geo.getsolarelevationmatrix(f.datetime(), sub_lon, lat, lon)
			solarelevation_min = solarelevation.min()
		return solarelevation_min

class CollectTimed(Process):
	class Meta:
        	app_label = 'plumbing'
	def do(self, stream):
		resultant_stream = {}
		for m in range(1,13,1):
			resultant_stream[m] = stream.clone()
		for fs in stream.files.all():
			fs.clone_for(resultant_stream[fs.file.datetime().month])
			fs.processed=True
			fs.save()
		return [v for k,v in resultant_stream]

class CollectChannel(Process):
	class Meta:
        	app_label = 'plumbing'
	channels = models.ManyToManyField(Channel,db_index=True)
	def do(self, stream):
		resultant_stream = {}
		for ch in channels:
			resultant_stream[ch.in_file] = stream.clone()
		for fs in stream.files.all():
			fs.clone_for(resultant_stream[fs.file.channel()])
			fs.processed=True
			fs.save()
		return [v for k,v in resultant_stream]

class FilterChannel(Process):
	class Meta:
        	app_label = 'plumbing'
	channels = models.ManyToManyField(Channel,db_index=True)
	def do(self, stream):
		resultant_stream = stream.clone()
		chs = [ str(ch.in_file) for ch in self.channels.all() ]
		sat = self.channels.all()[0].satellite
		for fs in stream.files.all():
			if fs.file.channel() in chs and fs.file.satellite() == sat.in_file:
				fs.clone_for(resultant_stream)
			fs.processed=True
			fs.save()
		return resultant_stream

class FilterTimed(Process):
	class Meta:
        	app_label = 'plumbing'
	time_range = models.ManyToManyField(UTCTimeRange,db_index=True)
	def do(self, srteam):
		resultant_stream = stream.clone()
		for fs in stream.files.all():
			if self.time_range.contains(fs.file.datetime()):
				fs.clone_for(resultant_stream)
			fs.processed=True
			fs.save()
		return resultant_stream

class AppendCountToRadiationCoefficient(Process):
	class Meta:
		app_label = 'plumbing'
	counts_shift = models.IntegerField()
	calibrated_coefficient = models.DecimalField(max_digits=5,decimal_places=3)
	space_measurement = models.DecimalField(max_digits=5,decimal_places=3)
	def do(self, stream):
		resultant_stream = stream.clone()
		for fs in stream.files.all():
			if f.channel() == '01':
				root, is_new = nc.open(f.completepath())
				nc.getdim(root,'coefficient',1)
				var = nc.getvar(root, 'counts_shift', 'f4', ('coefficient',), 4)
				var[0] = self.counts_shift
				var = nc.getvar(root, 'calibrated_coefficient', 'f4', ('coefficient',), 4)
				var[0] = self.calibrated_coefficient
				var = nc.getvar(root, 'space_measurement', 'f4', ('coefficient',), 4)
				var[0] = self.space_measurement
				#data = np.float32(self.calibrated_coefficient) * ((data / np.float32(self.counts_shift)) - np.float32(self.space_measurement))
				nc.close(root)
			fs.processed=True
			fs.save()
		return stream
