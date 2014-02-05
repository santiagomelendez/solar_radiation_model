# -*- coding: utf-8 -*- 
from plumbing.models import *
from django.test import TestCase


class TestProcessesOrder(TestCase):
	fixtures = [ 'initial_data.yaml', '*']

	def setUp(self):
		self.process_order = ProcessOrder.objects.get(pk=1)
		self.other_process_order = ProcessOrder.objects.get(pk=2)

	def test_serialization(self):
		# check if the __str__ method return the process __str__.
		self.assertEquals(str(self.process_order), str(unicode(self.process_order.process)))
		self.assertEquals(unicode(self.other_process_order), unicode(self.other_process_order.process))
		# check if the __unicode__ method is defined to return the process __unicode__.
		self.assertEquals(unicode(self.process_order), unicode(self.process_order.process))
		self.assertEquals(unicode(self.other_process_order), unicode(self.other_process_order.process))