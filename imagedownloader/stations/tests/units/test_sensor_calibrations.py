# -*- coding: utf-8 -*- 
from stations.models import *
from django.test import TestCase
from datetime import datetime
import pytz


class TestSensorCalibrations(TestCase):
	fixtures = [ 'initial_data.yaml', '*']
	
	def setUp(self):
		self.calibration = SensorCalibration.objects.filter(pk = 0)[0]

	def test_serialization(self):
		# check if the __str__ method is defined to return the object name.
		result = u'%2f x + %2f' % (self.calibration.coefficient, self.calibration.shift)
		self.assertEquals(str(self.calibration), result.encode("utf-8"))
		# check if the __unicode__ method is defined to return the object name.
		self.assertEquals(unicode(self.calibration), result)
