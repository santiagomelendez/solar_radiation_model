# -*- coding: utf-8 -*- 
from plumbing.models import *
from django.test import TestCase
from datetime import datetime
import pytz
import random


class TestAdapters(TestCase):
	fixtures = [ 'initial_data.yaml', '*']

	def setUp(self):
		self.stream = Stream(root_path="/var/service/data/GVAR_IMG/argentina/")
		self.stream.save()
		self.adapter = Adapter.objects.get_or_create(name='abstract one', stream=self.stream)[0]

	def test_update(self):
		# check if the update method raise a "Subclass responsability" exception because the subclass
		# should implement the method update.
		with self.assertRaises(Exception) as err:
			self.adapter.update()
		self.assertEquals(unicode(err.exception), u"Subclass responsability")