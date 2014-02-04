# -*- coding: utf-8 -*- 
from plumbing.models import *
from django.test import TestCase
import re


class TestMaterials(TestCase):
	fixtures = [ 'initial_data.yaml', '*']

	def setUp(self):
		self.material = Material()
		self.material.save()
		self.other_material = Material()
		self.other_material.save()

	def test_serialization(self):
		material = u'(created: %s, modified: %s)' % (unicode(self.material.created), unicode(self.material.modified))
		other_material = u'(created: %s, modified: %s)' % (unicode(self.other_material.created), unicode(self.other_material.modified))
		# check if the __str__ method return the created and modified datetime.
		self.assertEquals(str(self.material), str(material))
		self.assertEquals(str(self.other_material), str(other_material))
		# check if the __unicode__ method is defined to return the created and modified datetime.
		self.assertEquals(unicode(self.material), material)
		self.assertEquals(unicode(self.other_material), other_material)