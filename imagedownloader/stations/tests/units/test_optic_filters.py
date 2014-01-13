# -*- coding: utf-8 -*- 
from stations.models import *
from django.test import TestCase
from datetime import datetime
import pytz


class TestOpticFilters(TestCase):
	fixtures = [ 'initial_data.yaml', '*']
	
	def setUp(self):
		self.optic_filter = OpticFilter.objects.filter(name = 'No filtered')[0]

	def test_serialization(self):
		# check if the __str__ method is defined to return the object name as a string of bytes.
		self.assertEquals(str(self.optic_filter), self.optic_filter.name.encode("utf-8"))
		# check if the __unicode__ method is defined to return the string of bytes as a text.
		self.assertEquals(unicode(self.optic_filter), self.optic_filter.name)