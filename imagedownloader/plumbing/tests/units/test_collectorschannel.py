# -*- coding: utf-8 -*- 
from plumbing.models import *
from django.test import TestCase
from datetime import datetime
import pytz
import random
from copy import copy


class TestCollectorsChannel(TestCase):
	fixtures = [ 'initial_data.yaml', '*']

	def setUp(self):
		self.collect = CollectChannel.objects.filter(id=5)[0]
		self.root_path="/var/service/data/GVAR_IMG/argentina/"
		self.stream = Stream()
		self.stream.save()
		self.channels_str = [ "BAND_%s" % str(ch).zfill(2) for ch in range(1,7) ]
		random.shuffle(self.channels_str)
		self.files = [ File.objects.get_or_create(localname="%s2013/goes13.2013.M01.BAND_%s.nc" % (self.root_path, ch))[0] for ch in self.channels_str]
		for i in range(len(self.files)):
			self.files[i].save()
			fs = MaterialStatus.objects.get_or_create(material=self.files[i],stream=self.stream,processed=(i%2==0))[0]
			fs.save()

	def test_mark_with_tags(self):
		# check if the mark_with_tags method in the CollectChannel class don't
		# append a new tag into the stream.
		self.assertTrue(self.stream.tags.empty())
		self.collect.mark_with_tags(self.stream)
		self.assertTrue(self.stream.tags.empty())

	def test_get_key(self):
		# check if the class return the classification key of the file.
		for fs in self.stream.materials.all():
			self.assertEquals(self.collect.get_key(fs), "BAND_%s" % str(fs.material.channel()).zfill(2))