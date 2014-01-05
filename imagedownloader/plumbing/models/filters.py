from django.db import models
from requester.models import *
from core import *
import numpy as np
from libs.geometry import jaen as geo
from libs.file import netcdf as nc
import re
import sys
sys.path.append(".")


class FilterSolarElevation(Process):
	class Meta:
        	app_label = 'plumbing'
	minimum = models.DecimalField(max_digits=4,decimal_places=2)
	hourly_longitude = models.DecimalField(max_digits=5, decimal_places=2, default=Decimal('0.00'))

	def do(self,stream):
		resultant_stream = stream.clone()
		for fs in stream.files.all():
			sub_lon = np.float(self.hourly_longitude)
			lat,lon = fs.file.latlon()
			lower_solar_elevation = np.float(self.minimum)
			if self.solar_elevation(fs.file, sub_lon, lat, lon) >= lower_solar_elevation:
				fs.clone_for(resultant_stream)
		return resultant_stream

	def solar_elevation(self, f, sub_lon, lat, lon):
		solarelevation_min = -90.0
		if not lat is None:
			solarelevation = geo.getsolarelevationmatrix(f.datetime(), sub_lon, lat, lon)
			solarelevation_min = solarelevation.min()
		return solarelevation_min

	def mark_with_tags(self, stream):
		stream.tags.append("SE"+str(self.minimum))


class Collect(Process):
	class Meta:
		app_label = 'plumbing'

	def get_key(self, file_status):
		raise AttributeError("'Collect' object has no attribute 'get_key'")

	def get_keys(self, stream):
		return set([ self.get_key(fs) for fs in stream.files.all() ])

	def init_empty_streams(self, stream):
		keys = self.get_keys(stream)
		resultant_stream = {}
		for k in keys:
			resultant_stream[k] = stream.clone()
			resultant_stream[k].tags.insert_first(stream.files.all()[0].file.satellite())
		return resultant_stream

	def do(self, stream):
		resultant_stream = self.init_empty_streams(stream)
		for fs in stream.files.all():
			fs.clone_for(resultant_stream[self.get_key(fs)])
			fs.processed=True
			fs.save()
		for k in resultant_stream.keys():
			resultant_stream[k].tags.append(k)
		return resultant_stream.values()

	def mark_with_tags(self, stream):
		# Don't used because these process always return multiple streams
		pass


class CollectTimed(Collect):
	class Meta:
		app_label = 'plumbing'
	yearly = models.BooleanField()
	monthly = models.BooleanField()
	weekly = models.BooleanField()

	def get_key(self, file_status):
		r = ""
		dt = file_status.file.datetime()
		if self.yearly: r += str(dt.year)
		if self.monthly: r += (".M" + str(dt.month).zfill(2))
		if self.weekly: r += (".W" + str(dt.isocalendar()[1]).zfill(2))
		return r


class CollectChannel(Collect):
	class Meta:
		app_label = 'plumbing'

	def get_key(self, file_status):
		return "BAND_"+str(file_status.file.channel()).zfill(2)


class FilterChannel(Process):
	class Meta:
		app_label = 'plumbing'
	channels = models.ManyToManyField(Channel,db_index=True)

	def do(self, stream):
		resultant_stream = stream.clone()
		channels = self.channels.all()
		chs = [ str(ch.in_file) for ch in channels ]
		sat = channels[0].satellite
		for fs in stream.files.all():
			if fs.file.channel() in chs and fs.file.satellite() == sat.in_file:
				fs.clone_for(resultant_stream)
			fs.processed=True
			fs.save()
		return resultant_stream

	def mark_with_tags(self, stream):
		# Don't used because these process is transparent in the name
		pass


class FilterTimed(Process):
	class Meta:
		app_label = 'plumbing'
	time_range = models.ManyToManyField(UTCTimeRange,db_index=True)

	def do(self, stream):
		resultant_stream = stream.clone()
		for fs in stream.files.all():
			if self.time_range.contains(fs.file.datetime()):
				fs.clone_for(resultant_stream)
			fs.processed=True
			fs.save()
		return resultant_stream

	def mark_with_tags(self, stream):
		# TODO: Correct self.time_range.all() = self.time_range.begin self.time_range.end
		dates = [str(ch.in_file).zfill(2) for t in self.time_range.all()].join("_")
		stream.tags.append("UTC_"+re.sub("[\-\:\.\ ]","", str(d)))


class AppendCountToRadiationCoefficient(Process):
	class Meta:
		app_label = 'plumbing'

	def do(self, stream):
		from libs.sat.goes import calibration
		resultant_stream = stream.clone()
		for fs in stream.files.all():
			f = fs.file
			if f.channel() == '01':
				sat = f.satellite()
				root, is_new = nc.open(f.completepath())
				nc.getdim(root,'coefficient',1)
				var = nc.getvar(root, 'counts_shift', 'f4', ('coefficient',), 4)
				var[0] = calibration.counts_shift.coefficient(sat)
				var = nc.getvar(root, 'space_measurement', 'f4', ('coefficient',), 4)
				var[0] = calibration.space_measurement.coefficient(sat)
				var = nc.getvar(root, 'prelaunch', 'f4', ('coefficient',), 4)
				var[0] = calibration.prelaunch.coefficient(sat)[0]
				var = nc.getvar(root, 'postlaunch', 'f4', ('coefficient',), 4)
				dt = f.datetime()
				var[0] = calibration.postlaunch.coefficient(sat, dt.year, dt.month)
				#data = np.float32(self.calibrated_coefficient) * ((data / np.float32(self.counts_shift)) - np.float32(self.space_measurement))
				nc.close(root)
			fs.processed=True
			fs.save()
		return resultant_stream

	def mark_with_tags(self, stream):
		stream.tags.append("calibrated")