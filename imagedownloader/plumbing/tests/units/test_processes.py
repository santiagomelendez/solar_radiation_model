# -*- coding: utf-8 -*- 
from plumbing.models import *
from django.test import TestCase
from datetime import datetime
import pytz


class TestProcesses(TestCase):
	fixtures = [ 'initial_data.yaml', '*']

	def setUp(self):
		self.process = Process.objects.get_or_create(name='year.Mmonth')[0]
		self.other_process = Process.objects.get(pk=6)
		self.stream = Stream(root_path="/var/service/data/GVAR_IMG/argentina/")

	def test_serialization(self):
		# check if the __str__ method is defined to return the class name with the name parameter.
		self.assertEquals(str(self.process.type_cast()), 'CollectTimed [year.Mmonth]')
		self.assertEquals(str(self.other_process.type_cast()), "FilterSolarElevation [Filter night's images]")

	def test_mark_with_tags(self):
		# check if the mark_with_tags method in the Process class don't
		# append a new tag into the stream.
		self.process.mark_with_tags(self.stream)
		self.assertTrue(self.stream.empty())
		self.other_process.mark_with_tags(self.stream)
		self.assertTrue(self.stream.empty())