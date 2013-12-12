# -*- coding: utf-8 -*- 
from stations.models import *
from django.test import TestCase
from datetime import datetime
import pytz

class TestDevices(TestCase):
	fixtures = [ 'initial_data.yaml', '*']
	
	def setUp(self):
		self.device = Device.objects.filter(product__name = 'CMP 11')[0]

	def test_serialization(self):
		# check if the __str__ method is defined to return the object serial_number and a device product name.
		self.assertEquals(str(self.device), self.device.serial_number + " (" + str(self.device.product) + ")")