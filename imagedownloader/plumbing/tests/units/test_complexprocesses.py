# -*- coding: utf-8 -*- 
from plumbing.models import *
from django.test import TestCase
from datetime import datetime
import pytz


class TestComplexProcesses(TestCase):
	fixtures = [ 'initial_data.yaml', '*']

	def setUp(self):
		self.complex_process = ComplexProcess.objects.get_or_create(name='Filter nights and compact')[0]
		self.stream = Stream(root_path="/var/service/data/GVAR_IMG/argentina/")
		self.stream.save()

	def test_encapsulate_in_array(self):
		# check if the __str__ method is defined to return the object root_path parameter.
		self.assertEquals(self.complex_process.encapsulate_in_array(self.stream), [self.stream])
		self.assertEquals(self.complex_process.encapsulate_in_array([self.stream]), [self.stream])
