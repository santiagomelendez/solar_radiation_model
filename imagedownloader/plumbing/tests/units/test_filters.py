# -*- coding: utf-8 -*- 
from plumbing.models import *
from django.test import TestCase
from datetime import datetime
import pytz
import random
import aspects


class TestFilters(TestCase):
	fixtures = [ 'initial_data.yaml', '*']

	def setUp(self):
		self.filter = Filter.objects.create(name='abstract one')
		self.other_filter = Filter.objects.create(name='abstract two')
		self.other_filter.should_be_cloned = lambda material_status: True
		self.stream = Stream(root_path="/var/service/data/GVAR_IMG/argentina/")
		self.stream.save()
		self.materials = [ Material() for i in range(5) ]
		for i in range(len(self.materials)):
			self.materials[i].save()
			ms = MaterialStatus.objects.get_or_create(material=self.materials[i],stream=self.stream)[0]
			ms.save()

	def test_mark_with_tags(self):
		# check if the mark_with_tags method in the Collect class don't
		# append a new tag into the stream.
		self.assertTrue(self.stream.tags.empty())
		self.filter.mark_with_tags(self.stream)
		self.other_filter.mark_with_tags(self.stream)
		self.assertTrue(self.stream.tags.empty())

	def test_should_be_cloned(self):
		# check if return false by default.
		for ms in self.stream.materials.all():
			self.assertEquals(self.filter.should_be_cloned(ms), False)
			self.assertEquals(self.other_filter.should_be_cloned(ms), True)

	def test_do(self):
		# check if all the material statuses of the stream are unprocessed.
		self.assertEquals(self.stream.materials.filter(processed=True).count(), 0)
		# check if it call to should_be_cloned for each material_status of the stream.
		self.materials = []
		def filter_wrap(*args):
			result = yield aspects.proceed(*args)
			self.materials.append(args[1].material)
			# force the cloning of all the materials
			yield aspects.return_stop(True)
		filters = [ self.filter.should_be_cloned ]
		aspects.with_wrap(filter_wrap, *filters)
		result = self.filter.do(self.stream)
		self.assertTrue(len(self.materials) > 0)
		# check if the filtered materials are contained in the input stream.
		for ms in self.stream.materials.all():
			self.assertTrue(ms.material in self.materials)
			self.materials.remove(ms.material)
		self.assertEquals(len(self.materials), 0)