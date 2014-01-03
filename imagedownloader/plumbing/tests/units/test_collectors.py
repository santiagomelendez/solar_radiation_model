# -*- coding: utf-8 -*- 
from plumbing.models import *
from django.test import TestCase
from datetime import datetime
import pytz
import random


class TestCollectors(TestCase):
	fixtures = [ 'initial_data.yaml', '*']

	def setUp(self):
		self.collect = Collect.objects.get(name='year.Mmonth')
		self.stream = Stream(root_path="/var/service/data/GVAR_IMG/argentina/")
		self.stream.save()
		months = range(1,13)
		random.shuffle(months)
		self.files = [ File.objects.get_or_create(localname="%s2013/goes13.2013.M%s.BAND.1.nc" % (self.stream.root_path, str(i).zfill(2)))[0] for i in months]
		for i in range(len(self.files)):
			self.files[i].save()
			fs = FileStatus.objects.get_or_create(file=self.files[i],stream=self.stream,processed=(i%2==0))[0]
			fs.save()

	def test_mark_with_tags(self):
		# check if the mark_with_tags method in the Collect class don't
		# append a new tag into the stream.
		self.assertTrue(self.stream.tags.empty())
		self.collect.mark_with_tags(self.stream)
		self.assertTrue(self.stream.tags.empty())