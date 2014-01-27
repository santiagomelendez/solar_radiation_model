from django.db import models
from requester.models import Channel
from decimal import Decimal
from core import Process
import numpy as np
from libs.geometry import jaen as geo
from libs.file import netcdf as nc
from libs.geometry.jaen import getslots


class Filter(Process):
	class Meta(object):
		app_label = 'plumbing'

	def should_be_cloned(self, material_status):
		return False

	def do(self, stream):
		resultant_stream = stream.clone()
		for fs in stream.files.all():
			if self.should_be_cloned(fs):
				fs.clone_for(resultant_stream)
			fs.processed=True
			fs.save()
		return resultant_stream

	def mark_with_tags(self, stream):
		# Don't used because these process is transparent in the name
		pass


class FilterChannel(Filter):
	class Meta(object):
		app_label = 'plumbing'
	channels = models.ManyToManyField(Channel,db_index=True)

	def should_be_cloned(self, material_status):
		channels = self.channels.all()
		chs = [ unicode(ch.in_file) for ch in channels ]
		sat = channels[0].satellite
		return material_status.material.channel() in chs and material_status.material.satellite() == sat.in_file


class FilterTimed(Filter):
	class Meta(object):
		app_label = 'plumbing'
	yearly = models.BooleanField()
	monthly = models.BooleanField()
	weekly = models.BooleanField()
	week_day = models.BooleanField()
	daily = models.BooleanField()
	hourly = models.BooleanField()
	slotly = models.BooleanField()
	slots_by_day = models.IntegerField(default = 1)
	number = models.IntegerField(default=0)
	error = models.IntegerField(default=0)

	def contains(self, number):
		half_error = self.error / 2
		return self.number - half_error <= self.number <= self.number + half_error

	def should_be_cloned(self, material_status):
		dt = material_status.file.datetime()
		if self.yearly: return self.contains(dt.year)
		if self.monthly: return self.contains(dt.month)
		if self.weekly: return self.contains(dt.isocalendar()[1])
		if self.week_day: return self.contains(dt.weekday())
		if self.daily: return self.contains(dt.timetuple().tm_yday)
		if self.hourly: return self.contains(dt.hour)
		if self.slotly: return self.contains(getslots(dt.hour,self.slots_by_day/24))


class FilterSolarElevation(Filter):
	class Meta(object):
			app_label = 'plumbing'
	minimum = models.DecimalField(max_digits=4,decimal_places=2)
	hourly_longitude = models.DecimalField(max_digits=5, decimal_places=2, default=Decimal('0.00'))

	def solar_elevation(self, f, sub_lon, lat, lon):
		solarelevation_min = -90.0
		if not lat is None:
			solarelevation = geo.getsolarelevationmatrix(f.datetime(), sub_lon, lat, lon)
			solarelevation_min = solarelevation.min()
		return solarelevation_min

	def should_be_cloned(self, material_status):
		sub_lon = np.float(self.hourly_longitude)
		lat,lon = material_status.file.latlon()
		lower_solar_elevation = np.float(self.minimum)
		return self.solar_elevation(material_status.material, sub_lon, lat, lon) >= lower_solar_elevation

	def mark_with_tags(self, stream):
		stream.tags.append("SE"+str(self.minimum))


class AppendCountToRadiationCoefficient(Process):
	class Meta(object):
		app_label = 'plumbing'

	def do(self, stream):
		from libs.sat.goes import calibration
		resultant_stream = stream.clone()
		for fs in stream.files.all():
			f = fs.file
			if f.channel() == '01':
				sat = f.satellite()
				root = nc.open(f.completepath())[0]
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