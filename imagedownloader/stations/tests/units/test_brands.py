# -*- coding: utf-8 -*- 
from stations.models import *
from django.test import TestCase
from datetime import datetime
import pytz


class TestBrands(TestCase):
	fixtures = [ 'initial_data.yaml', '*']
	
	def setUp(self):
		self.brand = Brand.objects.filter(name = 'Kip&Zonen')[0]

	def test_serialization(self):
		# check if the __str__ method is defined to return the object name.
		self.assertEquals(str(self.brand), self.brand.name)