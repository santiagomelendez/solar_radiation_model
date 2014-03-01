# -*- coding: utf-8 -*- 
from plumbing.models import *
from django.test import TestCase
from datetime import datetime
import pytz
import random
from copy import copy


class TestCollectorsTimed(TestCase):
	fixtures = [ 'initial_data.yaml', '*']

	def setUp(self):
		self.years_collect = CollectTimed.objects.filter(id=7)[0]
		self.months_collect = CollectTimed.objects.filter(id=4)[0]
		self.weeks_collect = CollectTimed.objects.filter(id=8)[0]
		self.root_path="/var/service/data/GVAR_IMG/argentina/"
		self.stream = Stream()
		self.stream.save()
		self.days_tuples = [ (str(y),str(d*10)) for y in range(2012,2016) for d in range(1,365/10) ]
		random.shuffle(self.days_tuples)
		self.files = [ File.objects.get_or_create(localname="%s%s/goes13.%s.%s.160927.BAND_01.nc" % (self.root_path,dt[0],dt[0],dt[1]))[0] for dt in self.days_tuples]
		for f in self.files:
			f.save()
			fs = MaterialStatus.objects.get_or_create(material=f,stream=self.stream,processed=False)[0]
			fs.save()

	def test_mark_with_tags(self):
		# check if the mark_with_tags method in the CollectTimed class don't
		# append a new tag into the stream.
		self.assertTrue(self.stream.tags.empty())
		self.years_collect.mark_with_tags(self.stream)
		self.months_collect.mark_with_tags(self.stream)
		self.weeks_collect.mark_with_tags(self.stream)
		self.assertTrue(self.stream.tags.empty())

	def test_get_key(self):
		# check if the class return the classification key of the file.
		for fs in self.stream.materials.all():
			dt = fs.material.datetime()
			self.assertEquals(self.years_collect.get_key(fs), str(dt.year))
			self.assertEquals(self.months_collect.get_key(fs), "%s.M%s" % (str(dt.year),str(dt.month).zfill(2)))
			self.assertEquals(self.weeks_collect.get_key(fs), "%s.W%s" % (str(dt.year),str(dt.isocalendar()[1]).zfill(2)))