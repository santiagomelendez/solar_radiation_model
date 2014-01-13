# -*- coding: utf-8 -*- 
from stations.models import *
from django.test import TestCase
from datetime import datetime
import pytz


class TestPositions(TestCase):
	fixtures = [ 'initial_data.yaml', '*']
	
	def setUp(self):
		self.position = Position.objects.filter(station__name = 'Luján')[0]

	def test_serialization(self):
		# check if the __str__ method is defined to return an ascii.
		self.assertEquals(str(self.position), 'Luj\xc3\xa1n (-34.588163, -59.062993)')
		# to finish, check if the __unicode__ method is defined.
		self.assertEquals(unicode(self.position), u'Luján (-34.588163, -59.062993)')

	def test_coordinates(self):
		# check if the coordinates method return a string (with serialized coordinates).
		c = self.position.coordinates()
		self.assertEquals(c.__class__, str)
		# check if the coordinates can be deserialized to tuples.
		p = eval(c)
		self.assertEquals(p.__class__, tuple)
		self.assertEquals(len(p), 2)
		# check if each element of the tuple is a float.
		e = set([x.__class__ for x in p])
		self.assertEquals(len(e), 1)
		self.assertEquals(list(e)[0], float)