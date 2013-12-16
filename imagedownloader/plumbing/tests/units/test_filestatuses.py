# -*- coding: utf-8 -*- 
from plumbing.models import *
from django.test import TestCase
from datetime import datetime
import pytz


class TestFileStatuses(TestCase):
	fixtures = [ 'initial_data.yaml', '*']

	def setUp(self):
		self.stream = Stream(root_path="/var/service/data/GVAR_IMG/argentina/")
		self.stream.save()
		self.stream.tags.append("to_be_cloned")
		self.stream.tags.append("to_be_tested")
		self.second_stream = Stream(root_path="/var/service/data/GVAR_IMG/argentina/")
		self.second_stream.save()
		self.file = File.objects.get_or_create(localname="/var/service/data/argentina/2013/pkg.goes15.2013.W40.BAND_02.nc")[0]
		self.file.save()
		self.file_status = FileStatus.objects.get_or_create(file=self.file,stream=self.stream)[0]
		self.file_status.save()

	def test_clone(self):
		# check if the clone method create a new file_status.
		clone = self.file_status.clone_for(self.second_stream)
		self.assertNotEquals(clone, self.file_status)
		# check if the cloned file_status has the second_stream and the same file object.
		self.assertEquals(self.file_status.stream, self.stream)
		self.assertEquals(clone.stream, self.second_stream)
		self.assertEquals(clone.file, self.file_status.file)