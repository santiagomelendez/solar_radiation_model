# -*- coding: utf-8 -*- 
from stations.models import *
from django.test import TestCase
from datetime import datetime
import pytz


class TestStations(TestCase):
	fixtures = [ 'initial_data.yaml', '*']
	
	def setUp(self):
		self.station = Station.objects.filter(name = 'Luján')[0]

	def test_serialization(self):
		# check if the __str__ method is defined to return an ascii.
		self.assertEquals(str(self.station),"Luj\xc3\xa1n")
		# to finish, check if the __unicode__ method is defined.
		self.assertEquals(unicode(self.station), u'Luján')

	def test_coordinates(self):
		# check if the coordinates method return a list of strings (with serialized coordinates).
		for c in self.station.coordinates():
			self.assertEquals(c.__class__, str)