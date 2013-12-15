# -*- coding: utf-8 -*- 
from plumbing.models import *
from django.test import TestCase
from datetime import datetime
import pytz
import glob
import random


class TestStreams(TestCase):
	fixtures = [ 'initial_data.yaml', '*']

	def setUp(self):
		self.begin = datetime.utcnow().replace(tzinfo=pytz.UTC)
		self.stream = Stream(root_path="/var/service/data/GVAR_IMG/argentina/")
		self.stream.save()
		self.end = datetime.utcnow().replace(tzinfo=pytz.UTC)
		months = range(1,13)
		random.shuffle(months)
		self.files = [ File.objects.get_or_create(localname="%s2013/goes13.2013.M%s.BAND.1.nc" % (self.stream.root_path, str(i).zfill(2)))[0] for i in months]
		for i in range(len(self.files)):
			self.files[i].save()
			fs = FileStatus.objects.get_or_create(file=self.files[i],stream=self.stream,processed=(i%2==0))[0]
			fs.save()

	def test_serialization(self):
		# check if the __str__ method is defined to return the object root_path parameter.
		self.assertEquals(str(self.stream), self.stream.root_path)

	def test_save(self):
		# check if hte instance was created between the begining and the ending of the setup.
		self.assertTrue(self.begin <= self.stream.created <= self.end)
		# check if the created and modified datetime are equals
		self.assertEquals(self.stream.created, self.stream.modified)
		# check if the modified datetime change when the objects is saved again.
		self.stream.save()
		self.assertTrue(self.stream.modified > self.stream.created)

	def test_get_stream_from_file(self):
		# check if can extract the stream's root_path from the filename.
		filename = self.stream.root_path + "2013/goes13.2013.M12.BAND_01.nc"
		self.assertEquals(Stream.get_stream_from_file(filename), self.stream.root_path)

	def test_clone(self):
		# check if the clone method create a new stream.
		self.stream.tags.append("to_be_cloned")
		self.stream.tags.append("to_be_tested")
		clone = self.stream.clone()
		self.assertNotEquals(clone, self.stream)
		# check if the cloned stream has the same root_path.
		self.assertEquals(clone.root_path, self.stream.root_path)
		# check if the cloned stream has all the tags
		self.assertNotEquals(clone.tags, self.stream.tags)
		self.assertEquals(clone.tags.list(), self.stream.tags.list())
		# check if the cloned stream is empty, and if the clone method avoid clone the files.
		self.assertEquals(self.stream.files.count(), 12)
		self.assertEquals(clone.files.count(), 0)

	def test_unprocessed(self):
		# check if return only the unprocessed files.
		for fs in self.stream.unprocessed():
			self.assertFalse(fs.processed)
			fs.delete()
		# check if return an empty list when it don't have pending files.
		self.assertEquals(self.stream.unprocessed().count(), 0)

	def test_empty(self):
		# check if return True when it has got pending files.
		self.assertFalse(self.stream.empty())
		for fs in self.stream.unprocessed():
			fs.delete()
		# check if return False when it hasn't got pending files.
		self.assertTrue(self.stream.empty())

	def test_sorted_files(self):
		# check if all the files where sorted by the file datetime.
		prev = None
		for fs in self.stream.sorted_files():
			if prev:
				self.assertTrue(prev.file.filename() <= fs.file.filename())
			prev = fs