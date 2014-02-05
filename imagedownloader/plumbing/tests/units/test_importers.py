# -*- coding: utf-8 -*- 
from plumbing.models import *
from django.test import TestCase
from datetime import datetime
import pytz
import random
import threading


class TestImporters(TestCase):
	fixtures = [ 'initial_data.yaml', '*']

	def setUp(self):
		self.stream = Stream(root_path="/var/service/data/GVAR_IMG/argentina/")
		self.stream.save()
		self.importer = Importer.objects.get_or_create(name='abstract one', stream=self.stream)[0]

	def test_setup_unloaded(self):
		# check if create a thread for each unloaded importer.
		self.actives = threading.activeCount()
		self.unloaded = [ i for i in Importer.objects.all() if not hasattr(i, 'thread') ]
		self.loaded = Importer.setup_unloaded()
		self.assertEquals(self.actives + len(self.unloaded), threading.activeCount())
		for i in self.loaded:
			i.thread.cancel()