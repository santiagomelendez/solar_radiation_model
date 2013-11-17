from django.db import models
from decimal import Decimal
from datetime import datetime, timedelta
import hashlib
import os

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
		return '%i %s %s' % (self.serial_number, self.optic_filter.name, self.product.name)

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
		return self.__unicode__()
	def __unicode__(self):
		return u'%s %s' % (self.station.name, self.coordinates())

class Station(models.Model):
	name = models.TextField(db_index=True)
	def __str__(self):
		return self.name
	def __unicode__(self):
		return self.name
	def coordinates(self):
		return [p.coordinates() for p in self.position_set.all()]

class Configuration(models.Model):
	begin = models.DateTimeField(default=datetime.now())
	end = models.DateTimeField(blank=True, null=True)
	position = models.ForeignKey(Position)
	devices = models.ManyToManyField('Device', related_name='configurations')
	calibration = models.ForeignKey(SensorCalibration)
	created = models.DateTimeField(editable=False,default=datetime.utcnow())
	modified = models.DateTimeField(default=datetime.utcnow())
	backup = models.TextField(default="")
	@classmethod
	def actives(klass):
		return klass.objects.filter(end__isnull=True)
	def receive_temporal_file(self, f):
		with open(f.name, 'wb+') as destination:
			for chunk in f.chunks():
				destination.write(chunk)
	def added_measurements(self, f):
		if f.name.split(".")[-1] in ["csv", "xls"]:
			return True
		return False
	def backup_file(self, f, end):
		self.receive_temporal_file(f)
		if self.added_measurements(f):
			self.backup = self.get_backup_filename(f.name, hr=True)
			os.rename(f.name, self.backup)
			self.end = end
			self.save()
		else:
			os.remove(f.name)
	def get_backup_filename(self, path, block_size=256*128, hr=False):
		md5 = hashlib.md5()
		with open(path,'rb') as f:
			for chunk in iter(lambda: f.read(block_size), b''):
				md5.update(chunk)
		head = md5.hexdigest() if hr else md5.digest()
		return "stations/backup/"+ head[:6] + path
	def save(self, *args, **kwargs):
		""" On save, update timestamps """
		if not self.id:
			self.created = datetime.today()
		self.modified = datetime.today()
		return super(Configuration, self).save(*args, **kwargs)
	def __str__(self):
		return str(self.__unicode__())
	def __unicode__(self):
		return u'%s | %s | %s' % (self.position, str(self.modified), self.calibration )

class Measurement(models.Model):
	mean = models.DecimalField(max_digits=5,decimal_places=2,default=Decimal(0.00))
	between = models.IntegerField(default=0)
	finish = models.DateTimeField(default=datetime.utcnow())
	refresh_presision = models.IntegerField(default=0)
	configuration = models.ForeignKey(Configuration)
