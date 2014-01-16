# -*- coding: utf-8 -*- 
from plumbing.models import *
from django.test import TestCase
from datetime import datetime
import pytz
import random


class TestCollectors(TestCase):
	fixtures = [ 'initial_data.yaml', '*']

	def setUp(self):
		self.collect = Collect.objects.create(name='abstract one')
		self.other_collect = Collect.objects.create(name='abstract two')
		self.other_collect.get_key = lambda file_status: "even" if (file_status.material.datetime()).month % 2 == 0 else "uneven"
		self.stream = Stream(root_path="/var/service/data/GVAR_IMG/argentina/")
		self.stream.save()
		months = range(1,13)
		random.shuffle(months)
		self.files = [ File.objects.get_or_create(localname="%s2013/goes13.2013.M%s.BAND.1.nc" % (self.stream.root_path, str(i).zfill(2)))[0] for i in months]
		for i in range(len(self.files)):
			self.files[i].save()
			fs = MaterialStatus.objects.get_or_create(material=self.files[i],stream=self.stream,processed=(i%2==0))[0]
			fs.save()

	def test_mark_with_tags(self):
		# check if the mark_with_tags method in the Collect class don't
		# append a new tag into the stream.
		self.assertTrue(self.stream.tags.empty())
		self.collect.mark_with_tags(self.stream)
		self.assertTrue(self.stream.tags.empty())

	def test_get_key(self):
		# check if the abstract class should raise an exception because these method doesn't
		# exist on an abstract class.
		with self.assertRaises(AttributeError) as err:
			self.collect.get_key(self.stream.materials.all()[0])
		self.assertEquals(err.exception.message, "'Collect' object has no attribute 'get_key'")

	def test_get_keys(self):
		# check if when this is sended to an abstract class should raise the same exception that
		# with get_key.
		with self.assertRaises(AttributeError) as err:
			self.collect.get_keys(self.stream)
		self.assertEquals(err.exception.message, "'Collect' object has no attribute 'get_key'")
		# check if when a fake get_key method exists, a set of uniques keys is returned.
		keys = self.other_collect.get_keys(self.stream)
		self.assertEquals(len(keys), 2)
		for key in keys:
			self.assertTrue(key in ["even", "uneven"])

	def test_init_empty_streams(self):
		# check if when this is sended to an abstract class should raise the same exception that
		# with get_key.
		with self.assertRaises(AttributeError) as err:
			self.collect.init_empty_streams(self.stream)
		self.assertEquals(err.exception.message, "'Collect' object has no attribute 'get_key'")
		# check if when a fake get_key method exists, a set of uniques keys is returned.
		streams = self.other_collect.init_empty_streams(self.stream)
		self.assertEquals(len(streams.keys()), 2)
		for key in streams.keys():
			self.assertTrue(key in ["even", "uneven"])
			self.assertTrue(streams[key].empty())

	def test_do(self):
		# check if all the files statuses of the stream are unprocessed.
		self.assertTrue(self.stream.materials.filter(processed=True).count(), 0)
		# check if collect files into two differents streams: even and uneven (with the fake
		# get_key).
		other = {"even": "uneven", "uneven": "even"}
		result = self.other_collect.do(self.stream)
		for fs in self.stream.materials.all():
			self.assertTrue(fs.processed)
			key = self.other_collect.get_key(fs)
			stream = [s for s in result if s.tags.exist(key)][0]
			other_stream = [s for s in result if s.tags.exist(other[key])][0]
			self.assertNotEquals(stream, [])
			self.assertTrue(fs.material in [ f.material for f in stream.materials.all()])
			self.assertFalse(fs.material in [ f.material for f in other_stream.materials.all()])