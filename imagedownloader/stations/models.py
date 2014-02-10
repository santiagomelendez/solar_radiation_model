from django.db import models
from polymorphic import PolymorphicModel, PolymorphicManager
from decimal import Decimal
from datetime import datetime
import pytz


class OpticFilter(models.Model):
	name = models.TextField(db_index=True)

	def __str__(self):
		return unicode(self).encode("utf-8")

	def __unicode__(self):
		return self.name


class Brand(models.Model):
	name = models.TextField(db_index=True)

	def __str__(self):
		return unicode(self).encode("utf-8")

	def __unicode__(self):
		return self.name


class Product(models.Model):
	brand = models.ForeignKey(Brand)
	name = models.TextField(db_index=True)
	specifications = models.TextField(db_index=True)

	def __str__(self):
		return unicode(self).encode("utf-8")

	def __unicode__(self):
		return self.name


class Device(PolymorphicModel):
	objects = PolymorphicManager()
	product = models.ForeignKey(Product)
	serial_number = models.TextField(db_index=True,default="")
	description = models.TextField(db_index=True,default="")

	def __str__(self):
		return unicode(Device.objects.get(id=self.id)).encode("utf-8")

	def __unicode__(self):
		return u'%s %s (%s)' % (self.__class__.__name__, unicode(self.product), self.serial_number)


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
	angle = models.DecimalField(max_digits=7,decimal_places=4,default=Decimal('0.00'))


class SensorCalibration(models.Model):
	sensor = models.ForeignKey(Sensor)
	coefficient = models.DecimalField(max_digits=10,decimal_places=7,default=Decimal('0.00'))
	shift = models.DecimalField(max_digits=10,decimal_places=7,default=Decimal('0.00'))

	def __str__(self):
		return unicode(self).encode("utf-8")

	def __unicode__(self):
		return u'%2f x + %2f' % (self.coefficient, self.shift)


class Position(models.Model):
	station = models.ForeignKey('Station',null=True,default=None)
	""" A centimeter-presision point """
	latitude = models.DecimalField(max_digits=10,decimal_places=7,default=Decimal('0.00'))
	longitude = models.DecimalField(max_digits=10,decimal_places=7,default=Decimal('0.00'))

	def coordinates(self):
		return '(%4f, %4f)' % (self.latitude, self.longitude)

	def __str__(self):
		return unicode(self).encode('utf-8')

	def __unicode__(self):
		return u'%s %s' % (self.station.name, self.coordinates())


class Station(models.Model):
	name = models.TextField(db_index=True)

	def __str__(self):
		return unicode(self).encode("utf-8")

	def __unicode__(self):
		return self.name

	def coordinates(self):
		return [p.coordinates() for p in self.position_set.all()]


class Configuration(models.Model,object):
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

	def append_rows(self, rows, between, refresh_presision):
		"""Transform the rows of data to Measurements.

			Keyword arguments:
			rows -- an array of arrays [datetime, integral_measurement]
			between -- time between integral_measurements in seconds
			refresh_presision -- time between sensor values that compose the integral_measurements
			"""
		for r in rows:
			Measurement.register_or_check(finish=r[0], mean=r[1]/between, between=between, refresh_presision=refresh_presision, configuration=self)

	def go_inactive(self, dt=datetime.utcnow().replace(tzinfo=pytz.UTC)):
		"""Make the configuration object inactive.

			Keyword arguments:
			dt -- datetime of the moment when the configuration go inactive
			"""
		self.end = dt
		self.save()

	def register_measurements(self, end, rows, between, refresh_presision):
		"""Register the measurements if it has measurements and close the configuration, if it hasen't got measurements clean the temporal file on disk.

			Keyword arguments:
			f -- open memory file
			end -- datetime of the moment when the configuration go inactive
			between -- time between integral_measurements in seconds
			refresh_presision -- time between sensor values that compose the integral_measurements
			"""
		if not self.end and len(rows) > 0:
			self.append_rows(rows, between, refresh_presision)
			self.go_inactive(end)
			self.save()

	def get_backup_filename(self, path):
		"""Proposes a name for the backup file.

			Keyword arguments:
			path -- temporal filename
			"""
		head = datetime.utcnow().replace(tzinfo=pytz.UTC).strftime("%Y%m%d%H%M%S")
		self.backup = "stations/backup/%s.%s" % (head, path)
		return self.backup

	def save(self, *args, **kwargs):
		""" On save, update timestamps """
		now = datetime.utcnow().replace(tzinfo=pytz.UTC)
		if not self.pk:
			self.created = now
		self.modified = now
		return super(Configuration, self).save(*args, **kwargs)

	def __str__(self):
		return unicode(self).encode('utf8')

	def __unicode__(self):
		return u'%s | %s | %s' % (self.position, unicode(self.modified), self.calibration )

class InvalidMeasurementError(RuntimeError):
	pass


class Measurement(models.Model):
	mean = models.DecimalField(max_digits=5,decimal_places=2,default=Decimal('0.00'))
	between = models.IntegerField(default=0)
	finish = models.DateTimeField(default=datetime.utcnow().replace(tzinfo=pytz.UTC))
	refresh_presision = models.IntegerField(default=0)
	configuration = models.ForeignKey(Configuration)
	class Meta(object):
		unique_together = ('configuration', 'finish',)

	@classmethod
	def register_or_check(klass, finish, mean, between, refresh_presision, configuration):
		"""Return the active configurations."""
		m, created = klass.objects.get_or_create(finish=finish, configuration=configuration)
		if created:
			m.mean=mean
			m.between=between
			m.refresh_presision=refresh_presision
			m.save()
		else:
			diff = abs(float(m.mean) - mean)
			if not(diff < 0.006 and m.between == between and m.refresh_presision == refresh_presision):
				raise InvalidMeasurementError("There are diferents values for the same measurement.")
		return m

	def __str__(self):
		return unicode(self).encode("utf-8")

	def __unicode__(self):
		return u'%s %s %.2f (%i sec)' % (unicode(self.configuration), unicode(self.finish), self.mean, self.between)