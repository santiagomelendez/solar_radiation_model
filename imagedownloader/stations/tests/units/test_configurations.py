# -*- coding: utf-8 -*- 
from stations.models import *
from django.test import TestCase
from datetime import datetime, timedelta
from decimal import Decimal
import pytz


class TestConfigurations(TestCase):
	fixtures = [ 'initial_data.yaml', '*']

	def setUp(self):
		self.conf = Configuration.objects.filter(position__station__name = 'Luján')[0]
		self.begin = datetime.utcnow().replace(tzinfo=pytz.UTC)
		self.configuration = Configuration(position=self.conf.position, calibration=self.conf.calibration)
		self.configuration.save()
		self.end = datetime.utcnow().replace(tzinfo=pytz.UTC)
		self.rows = [
			[datetime(2013,12,8,16,40,0).replace(tzinfo=pytz.UTC), Decimal('300.1')],
			[datetime(2013,12,8,16,50,0).replace(tzinfo=pytz.UTC), Decimal('200.1')],
			[datetime(2013,12,8,17,00,0).replace(tzinfo=pytz.UTC), Decimal('100.1')]
			]

	def test_save(self):
		# check if hte instance was created between the begining and the ending of the setup.
		self.assertTrue(self.begin <= self.configuration.created <= self.end)
		# check if the created and modified datetime are equals
		self.assertEquals(self.configuration.created, self.configuration.modified)
		# check if the modified datetime change when the objects is saved again
		self.configuration.save()
		self.assertTrue(self.configuration.modified > self.configuration.created)

	def test_serialization(self):
		# check if the __str__ method is defined to return the object position, when it was modified and the calibration parameters.
		result = u'%s | %s | %s' % (self.configuration.position, unicode(self.configuration.modified), self.configuration.calibration )
		self.assertEquals(str(self.configuration), result.encode("utf-8"))
		# check if the __unicode__ method is defined to return the string of bytes as a text.
		self.assertEquals(unicode(self.configuration), result)

	def test_go_inactive(self):
		# put the end attribute to None to make the configuration
		# active.
		self.configuration.end = None
		self.configuration.save()
		self.assertEqual(self.configuration.end, None)
		# then test that the method go_inactive set the end attribute.
		now = datetime.utcnow().replace(tzinfo=pytz.UTC)
		self.configuration.go_inactive(now)
		self.assertEqual(self.configuration.end, now)

	def test_actives(self):
		# make sure that returns the active configurations set,
		# that mean the configurations without an end attribute.
		are_active = Configuration.actives()
		self.assertTrue(len(are_active) > 0)
		# then if all the active configurations go_inactive, the
		# returned set should have 0 elements.
		for c in are_active:
			c.go_inactive()
		self.assertEqual(len(Configuration.actives()), 0)

	def test_get_backup_filename(self):
		# check if the backup-filename uses the
		# stations/backup/ folder.
		before = datetime.utcnow().replace(microsecond=0)
		filename = self.configuration.get_backup_filename("file.xls")
		after = datetime.utcnow().replace(microsecond=0)
		parts = filename.split(".")
		self.assertEqual(parts[0][:-14], "stations/backup/")
		# then test if the datetime added at the begining of the name use the UTC
		# clock (and check that is between 2 datetime captures).
		dt = datetime.strptime(parts[0][-14:],"%Y%m%d%H%M%S")
		self.assertTrue(before <= dt <= after)
		# to finish, test if the original name is at the end of the generated
		# backup-filename. 
		self.assertEqual(filename[-8:], "file.xls")

	def test_append_rows(self):
		# check if the configuration return an empty queryset when don't have
		# measurements.
		self.assertEqual(self.configuration.measurement_set.count(), 0)
		# check if the method append_rows add multiple measurements to
		# the configuration.
		self.configuration.append_rows(self.rows, 600, 1)
		self.assertEqual(self.configuration.measurement_set.count(), 3)
		# check if the measurement_set raise an exception when the application
		# try to remove from the queryset.
		measurements = self.configuration.measurement_set.all()
		with self.assertRaises(AttributeError):
			self.configuration.measurement_set.remove(measurements[0])
		# check if the configuration is updated when a measure is removed.
		measurements[0].delete()
		self.assertEqual(self.configuration.measurement_set.count(), 2)
		# to finish, remove all the measurements and check if it update the
		# configuration.
		for m in self.configuration.measurement_set.all():
			m.delete()
		self.assertEqual(self.configuration.measurement_set.count(), 0)

	def test_register_measurements(self):
		# check if the configuration is active and hasn't got measurements registered
		self.assertEquals(self.configuration.end, None)
		self.assertEquals(self.configuration.measurement_set.count(),0)
		# check if register the measurements and close the configuration
		now = datetime.utcnow().replace(tzinfo=pytz.UTC)
		self.configuration.register_measurements(now, self.rows, 600, 1)
		self.assertEquals(self.configuration.end, now)
		self.assertEquals(self.configuration.measurement_set.count(),len(self.rows))
