# -*- coding: utf-8 -*- 
from plumbing.models import *
from django.test import TestCase
from datetime import datetime
import pytz

class TestTagManagers(TestCase):
	fixtures = [ 'initial_data.yaml', '*']

	def setUp(self):
		self.tag_manager = TagManager()
		self.loaded_tag_manager = TagManager(tag_string="vegetable,food,red,spice")

	def test_list(self):
		# check if the list is empty.
		self.assertEquals(self.tag_manager.list(), [])
		# check if the list contains only the tag_manager tags.
		self.assertFalse("vegetable" in self.tag_manager.list())
		the_list = self.loaded_tag_manager.list()
		self.assertTrue("vegetable" in the_list)
		self.assertTrue("food" in the_list)
		self.assertTrue("red" in the_list)
		self.assertTrue("spice" in the_list)
		self.assertFalse("tomato" in the_list)
		self.assertFalse("" in the_list)

	def test_exist(self):
		# check if the empty string is avoided from tag_manager.
		self.assertFalse(self.tag_manager.exist(""))
		# check if the list contains all the created tags.
		self.assertTrue(self.loaded_tag_manager.exist("vegetable"))
		self.assertTrue(self.loaded_tag_manager.exist("food"))
		self.assertTrue(self.loaded_tag_manager.exist("red"))
		self.assertTrue(self.loaded_tag_manager.exist("spice"))
		self.assertFalse(self.loaded_tag_manager.exist("tomato"))
		self.assertFalse(self.loaded_tag_manager.exist(""))

	def test_clone(self):
		# check if the empty tag_manager is copied in other instance object.
		clone = self.tag_manager.clone()
		self.assertNotEquals(clone.id, self.tag_manager.id)
		self.assertEquals(clone.list(), self.tag_manager.list())
		# check if a loaded tag_manager is copied in other instace object.
		clone = self.loaded_tag_manager.clone()
		self.assertNotEquals(clone.id, self.tag_manager.id)
		self.assertEquals(clone.list(), self.loaded_tag_manager.list())

	def test_insert_first(self):
		# check if it insert the element at the begining of the empty list.
		self.assertFalse(self.tag_manager.exist("tomato"))
		self.tag_manager.insert_first("tomato")
		self.assertEquals(self.tag_manager.list()[0], "tomato")
		# check if the uniqueness of the element in the list fails the manager
		# should avoid the insertion if it exists.
		self.assertTrue(self.loaded_tag_manager.exist("red"))
		self.loaded_tag_manager.insert_first("red")
		self.assertEquals(self.loaded_tag_manager.list()[2], "red")
		self.assertNotEquals(self.loaded_tag_manager.list()[0], "red")
		# check if it insert the element at the begining of the loaded list.
		self.assertFalse(self.loaded_tag_manager.exist("tomato"))
		self.loaded_tag_manager.insert_first("tomato")
		self.assertEquals(self.loaded_tag_manager.list(), ["tomato","vegetable","food","red","spice"])

	def test_append(self):
		# check if it insert the element at the begining of the empty list.
		self.assertFalse(self.tag_manager.exist("tomato"))
		self.tag_manager.append("tomato")
		self.assertEquals(self.tag_manager.list()[-1], "tomato")
		# check if the uniqueness of the element in the list fails the manager
		# should avoid the insertion if it exists.
		self.assertTrue(self.loaded_tag_manager.exist("red"))
		self.loaded_tag_manager.append("red")
		self.assertEquals(self.loaded_tag_manager.list()[2], "red")
		self.assertNotEquals(self.loaded_tag_manager.list()[-1], "red")
		# check if it insert the element at the begining of the loaded list.
		self.assertFalse(self.loaded_tag_manager.exist("tomato"))
		self.loaded_tag_manager.append("tomato")
		self.assertEquals(self.loaded_tag_manager.list(), ["vegetable","food","red","spice","tomato"])

	def test_make_filename(self):
		# check if it return a string with the tags concatenated.
		tags = ["goes13","2011","M06","BAND_01"]
		for t in tags:
			self.tag_manager.append(t)
		self.assertEquals(self.tag_manager.make_filename(), "goes13.2011.M06.BAND_01")