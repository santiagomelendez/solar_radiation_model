# -*- coding: utf-8 -*- 
from plumbing.models import *
from django.test import TestCase
from datetime import datetime
import pytz


class TestMaterialStatuses(TestCase):
	fixtures = [ 'initial_data.yaml', '*']

	def setUp(self):
		self.stream = Stream(root_path="/var/service/data/GVAR_IMG/argentina/")
		self.stream.save()
		self.stream.tags.append("to_be_cloned")
		self.stream.tags.append("to_be_tested")
		self.second_stream = Stream(root_path="/var/service/data/GVAR_IMG/argentina/")
		self.second_stream.save()
		self.material = Material()
		self.material.save()
		self.material_status = MaterialStatus.objects.get_or_create(material=self.material,stream=self.stream)[0]
		self.material_status.save()

	def test_serialization(self):
		material_status = u'%s -> %s' % (unicode(self.material_status.stream), unicode(self.material_status.material))
		# check if the __str__ method return the created and modified datetime.
		self.assertEquals(str(self.material_status), str(material_status))
		# check if the __unicode__ method is defined to return the created and modified datetime.
		self.assertEquals(unicode(self.material_status), material_status)

	def test_clone(self):
		# check if the clone method create a new file_status.
		clone = self.material_status.clone_for(self.second_stream)
		self.assertNotEquals(clone, self.material_status)
		# check if the cloned file_status has the second_stream and the same file object.
		self.assertEquals(self.material_status.stream, self.stream)
		self.assertEquals(clone.stream, self.second_stream)
		self.assertEquals(clone.material, self.material_status.material)