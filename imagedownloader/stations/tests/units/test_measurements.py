# -*- coding: utf-8 -*- 
from stations.models import *
from django.test import TestCase
from datetime import datetime
import pytz


class TestMeasurements(TestCase):
	fixtures = [ 'initial_data.yaml', '*']

	def setUp(self):
		self.finish = datetime.utcnow().replace(tzinfo=pytz.UTC)
		self.mean = 60
		self.between = 600
		self.refresh_presision = 1
		self.configuration = Configuration.objects.filter(position__station__name = 'Luján')[0]

	def test_register_or_check(self):
		# check if that there aren't measurements.
		self.assertEquals(Measurement.objects.count(), 0)
		# check if the object is registed the first time.
		m1 = Measurement.register_or_check(self.finish, self.mean, self.between, self.refresh_presision, self.configuration)
		self.assertEquals(Measurement.objects.count(), 1)
		# check if when saving it again avoid creating a new measurement in the database
		# and return the same object.
		m2 = Measurement.register_or_check(self.finish, self.mean, self.between, self.refresh_presision, self.configuration)
		self.assertEquals(Measurement.objects.count(), 1)
		# check if when saving it again without the same mean, between or presision
		# raise an exception.
		with self.assertRaises(InvalidMeasurementError):
			m3 = Measurement.register_or_check(self.finish, self.mean + 1, self.between, self.refresh_presision, self.configuration)
		self.assertEquals(Measurement.objects.count(), 1)

	def test_serialization(self):
		# check if the __str__ method is defined to return the object configuration, finish, mean and between attributes.
		measurement = Measurement.register_or_check(self.finish, self.mean, self.between, self.refresh_presision, self.configuration)
		result = u'%s %s %.2f (%i sec)' % (unicode(measurement.configuration), unicode(measurement.finish), self.mean, self.between)
		self.assertEquals(str(measurement), result.encode("utf-8"))
		# check if the __unicode__ method is defined to return the string of bytes as a text.
		self.assertEquals(unicode(measurement), result)