# -*- coding: utf-8 -*- 
from stations.models import *
from django.test import TestCase
from datetime import datetime
import pytz


class TestSensors(TestCase):
	fixtures = [ 'initial_data.yaml', '*']
	
	def setUp(self):
		self.sensor = Sensor.objects.filter(product__name = 'CMP 11')[0]

	def test_sensor_pretty_name(self):
		# check if the sensor_pretty_name method is defined to return the object name.
		self.assertEquals(self.sensor.sensor_pretty_name(), self.sensor.serial_number + " " + self.sensor.optic_filter.name + " " + self.sensor.product.name)