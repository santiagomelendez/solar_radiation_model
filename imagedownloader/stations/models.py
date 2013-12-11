from django.db import models
from decimal import Decimal
from datetime import datetime, timedelta
import pytz
import hashlib
import os
from importer import from_csv, from_xls

class OpticFilter(models.Model):
	name = models.TextField(db_index=True)
	def __str__(self):
		return self.name

class Brand(models.Model):
	name = models.TextField(db_index=True)
	def __str__(self):
		return self.name

class Product(models.Model):
	brand = models.ForeignKey(Brand)
	name = models.TextField(db_index=True)
	specifications = models.TextField(db_index=True)
	def __str__(self):
		return self.name

class Device(models.Model):
	product = models.ForeignKey(Product)
	serial_number = models.TextField(db_index=True,default="")
	description = models.TextField(db_index=True,default="")
	def __str__(self):
		return "%s (%s)" % (self.serial_number, self.product)

class Sensor(Device):
	optic_filter = models.ForeignKey(OpticFilter,null=True)
	def sensor_pretty_name(self):
		return '%s %s %s' % (self.serial_number, self.optic_filter, self.product.name)

class Datalogger(Device):
	pass

class Tracker(Device):
	pass

class ShadowBall(Device):
	pass

class InclinedSupport(Device):
	angle = models.DecimalField(max_digits=7,decimal_places=4,default=Decimal(0.00))

class SensorCalibration(models.Model):
	sensor = models.ForeignKey(Sensor)
	coefficient = models.DecimalField(max_digits=10,decimal_places=7,default=Decimal(0.00))
	shift = models.DecimalField(max_digits=10,decimal_places=7,default=Decimal(0.00))
	def __str__(self):
		return '%2f x + %2f' % (self.coefficient, self.shift)

class Position(models.Model):
	station = models.ForeignKey('Station',null=True,default=None)
	""" A centimeter-presision point """
	latitude = models.DecimalField(max_digits=10,decimal_places=7,default=Decimal(0.00))
	longitude = models.DecimalField(max_digits=10,decimal_places=7,default=Decimal(0.00))
	def coordinates(self):
		return '(%4f, %4f)' % (self.latitude, self.longitude)
	def __str__(self):
		return self.__unicode__().encode('ascii', 'ignore')
	def __unicode__(self):
		return u'%s %s' % (self.station.name, self.coordinates())

class Station(models.Model):
	name = models.TextField(db_index=True)
	def __str__(self):
		return self.name.encode('ascii', 'ignore')
	def __unicode__(self):
		return self.name
	def coordinates(self):
		return [p.coordinates() for p in self.position_set.all()]

class Configuration(models.Model):
	begin = models.DateTimeField(default=datetime.utcnow().replace(tzinfo=pytz.UTC))
	end = models.DateTimeField(blank=True, null=True)
	position = models.ForeignKey(Position)
	devices = models.ManyToManyField('Device', related_name='configurations')
	calibration = models.ForeignKey(SensorCalibration)
	created = models.DateTimeField(editable=False,default=datetime.utcnow().replace(tzinfo=pytz.UTC))
	modified = models.DateTimeField(default=datetime.utcnow().replace(tzinfo=pytz.UTC))
	backup = models.TextField(default="")
	@classmethod
	def actives(klass):
		"""Return the active configurations."""
		return klass.objects.filter(end__isnull=True)
	def receive_temporal_file(self, f):
		"""Save an open memory file in a temporal disk file.

		    Keyword arguments:
		    f -- open memory file
		    """
		with open(f.name, 'wb+') as destination:
			for chunk in f.chunks():
				destination.write(chunk)
	def transform_csv_measurements(self, filename):
		utc_diff = -3
		timestamp_col = 0
		channel = 1
		skip_rows = 3
		return from_csv(filename, utc_diff, timestamp_col, channel, skip_rows)
	def transform_xls_measurements(self, filename):
		utc_diff = -3
		i_sheet = 1
		x_year = 1
		x_julian = 2
		x_timestamp = 3
		x_value = 9
		y_from = 10
		return from_xls(filename, utc_diff, i_sheet, x_year, x_julian, x_timestamp, x_value, y_from)
	def append_rows(self, rows, between, refresh_presision):
		"""Transform the rows of data to Measurements.

		    Keyword arguments:
		    rows -- an array of arrays [datetime, integral_measurement]
			between -- time between integral_measurements in seconds
			refresh_presision -- time between sensor values that compose the integral_measurements
		    """
		for r in rows:
			m = Measurement(mean=r[1]/between, between=between, finish=r[0], refresh_presision=refresh_presision, configuration=self)
			m.save()
	def added_measurements(self, f, between, refresh_presision):
		"""	Transform the files (xls, csv) to measurements.

		    Keyword arguments:
		    f -- open memory file
			between -- time between integral_measurements in seconds
			refresh_presision -- time between sensor values that compose the integral_measurements
		    """
		extension = f.name.split(".")[-1]
		if extension in ["csv", "xls"]:
			rows = getattr(self, "transform_%s_measurements" % extension)(f.name)
			self.append_rows(rows, between, refresh_presision)
			return True
		return False
	def go_inactive(self, dt=datetime.utcnow().replace(tzinfo=pytz.UTC)):
		"""Make the configuration object inactive.

		    Keyword arguments:
		    dt -- datetime of the moment when the configuration go inactive
		    """
		self.end = dt
		self.save()
	def backup_file(self, f, end, between, refresh_presision):
		"""Backup the file if it has measurements and clise the configuration, if clean the temporal file on disk.

		    Keyword arguments:
		    f -- open memory file
			end -- datetime of the moment when the configuration go inactive
			between -- time between integral_measurements in seconds
			refresh_presision -- time between sensor values that compose the integral_measurements
		    """
		self.receive_temporal_file(f)
		if self.added_measurements(f, between, refresh_presision):
			self.backup = self.get_backup_filename(f.name)
			os.rename(f.name, self.backup)
			self.go_inactive(end)
			self.save()
		else:
			os.remove(f.name)
	def get_backup_filename(self, path):
		"""Proposes a name for the backup file.

		    Keyword arguments:
		    path -- temporal filename
		    """
		head = datetime.utcnow().replace(tzinfo=pytz.UTC).strftime("%Y%m%d%H%M%S")
		return "stations/backup/%s.%s" % (head, path)
	def save(self, *args, **kwargs):
		""" On save, update timestamps """
		now = datetime.utcnow().replace(tzinfo=pytz.UTC)
		if not self.id:
			self.created = now
		self.modified = now
		return super(Configuration, self).save(*args, **kwargs)
	def __str__(self):
		return self.__unicode__().encode('ascii', 'ignore')
	def __unicode__(self):
		return u'%s | %s | %s' % (self.position, str(self.modified), self.calibration )

class Measurement(models.Model):
	mean = models.DecimalField(max_digits=5,decimal_places=2,default=Decimal(0.00))
	between = models.IntegerField(default=0)
	finish = models.DateTimeField(default=datetime.utcnow().replace(tzinfo=pytz.UTC))
	refresh_presision = models.IntegerField(default=0)
	configuration = models.ForeignKey(Configuration)
	class Meta:
		unique_together = ('configuration', 'finish',)
