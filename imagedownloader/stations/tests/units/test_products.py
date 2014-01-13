# -*- coding: utf-8 -*- 
from stations.models import *
from django.test import TestCase
from datetime import datetime
import pytz


class TestProducts(TestCase):
	fixtures = [ 'initial_data.yaml', '*']
	
	def setUp(self):
		self.product = Product.objects.filter(name = 'CMP 11')[0]

	def test_serialization(self):
		# check if the __str__ method is defined to return the object name.
		self.assertEquals(str(self.product), self.product.name.encode("utf-8"))
		# check if the __unicode__ method is defined to return the string of bytes as a text.
		self.assertEquals(unicode(self.product), self.product.name)