# -*- coding: utf-8 -*- 
from requester.models import File
from django.test import TestCase
from datetime import datetime
import pytz


class TestFiles(TestCase):
	fixtures = [ 'initial_data.yaml', '*']

	def setUp(self):
		months = range(1,13)
		self.file = File.objects.get_or_create(localname="/var/service/data/argentina/2013/pkg.goes15.2013.W40.BAND_02.nc")[0]
		self.file.save()
		self.other_file = File.objects.get_or_create(localname="/var/service/data/argentina/2013/pkg.goes13.2013.M12.BAND_01.nc")[0]
		self.other_file.save()
		self.simple_file = File.objects.get_or_create(localname="/var/service/data/argentina/2013/goes13.2013.100.231032.BAND_04.nc")[0]
		self.simple_file.save()

	def test_serialization(self):
		# check if the __str__ method return only the filename without the path.
		self.assertEquals(str(self.file), "pkg.goes15.2013.W40.BAND_02.nc")

	def test_comparisons(self):
		# check if the __gt__ method return True if a file has a datetime greater than other_file.
		self.assertTrue(self.other_file > self.file)
		self.assertTrue(self.file > self.simple_file)
		# check if the __lt__ method return True if other_file has a datetime lower than a file.
		self.assertTrue(self.file < self.other_file)
		self.assertTrue(self.simple_file < self.file)
		# check if both files are equals it return false.
		self.file.localname = self.other_file.localname
		self.simple_file.localname = self.other_file.localname
		self.assertFalse(self.other_file > self.file)
		self.assertFalse(self.file < self.other_file)
		self.assertFalse(self.file > self.simple_file)
		self.assertFalse(self.simple_file < self.file)

	def test_channel(self):
		# check if the channel method return a string identifying a satellite channel.
		self.assertEquals(self.file.channel(), '02')
		self.assertEquals(self.other_file.channel(), '01')

	def test_satellite(self):
		# check if the satellite method return a string identifying a unique satellite.
		self.assertEquals(self.file.satellite(), "goes15")
		self.assertEquals(self.other_file.satellite(), "goes13")

	def test_deduce_year_fraction(self):
		# check if the deduced datetime correspond is right.
		files = [self.file, self.other_file, self.simple_file]
		results = [datetime(2013,10,07,00,00,00), datetime(2013,12,31,00,00,00), datetime(2013,04,10,23,10,32)]
		for i in range(len(files)):
			f_info = files[i].filename().replace('pkg.','').split(".")
			self.assertEquals(files[i].deduce_year_fraction(int(f_info[1]),f_info[2:]), results[i])