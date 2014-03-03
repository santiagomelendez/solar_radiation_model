from django.db import models
from polymorphic import PolymorphicModel, PolymorphicManager
from datetime import datetime, timedelta
import pytz  # 3rd party
import os
from django.db.models import Min, Max
from decimal import Decimal
from netCDF4 import Dataset
from libs.console import total_seconds
from materials import Request, Order

# Create your models here.

class Area(models.Model):
	class Meta(object):
		app_label = 'requester'
	name = models.TextField()
	north_latitude = models.DecimalField(max_digits=4,decimal_places=2)
	south_latitude = models.DecimalField(max_digits=4,decimal_places=2)
	east_longitude = models.DecimalField(max_digits=5,decimal_places=2)
	west_longitude = models.DecimalField(max_digits=5,decimal_places=2)
	hourly_longitude = models.DecimalField(max_digits=5, decimal_places=2, default=Decimal('0.00'))
	created = models.DateTimeField(auto_now_add=True)
	modified = models.DateTimeField(auto_now=True)

	def __str__(self):
		return unicode(self).encode("utf-8")

	def __unicode__(self):
		return self.name

	def getlongitude(self,datetime=datetime.utcnow().replace(tzinfo=pytz.UTC)):
		return self.hourly_longitude


class UTCTimeRange(models.Model):
	class Meta(object):
		app_label = 'requester'
	begin = models.DateTimeField()
	end = models.DateTimeField()
	created = models.DateTimeField(auto_now_add=True)
	modified = models.DateTimeField(auto_now=True)

	def __str__(self):
		return unicode(self).encode("utf-8")

	def __unicode__(self):
		return u'%s -> %s' % (unicode(self.begin), unicode(self.end))

	def step(self):
		begin = datetime.utcnow().replace(tzinfo=pytz.UTC) if self.begin is None else self.begin
		end = datetime.utcnow().replace(tzinfo=pytz.UTC) if self.end is None else self.end
		diff = total_seconds(end - begin)
		return timedelta(days=(diff / abs(diff)))

	def steps(self):
		return int(total_seconds(self.end - self.begin) / total_seconds(self.step()))

	def contains(self, dt):
		timezoned = dt.replace(tzinfo=pytz.UTC)
		return self.begin <= timezoned and self.end >= timezoned


class Account(PolymorphicModel,object):
	class Meta(object):
		app_label = 'requester'
	objects = PolymorphicManager()
	password = models.TextField()
	created = models.DateTimeField(auto_now_add=True)
	modified = models.DateTimeField(auto_now=True)

	def identification(self):
		return u'%s, %s' % (self.password, unicode(self.created))

	def __str__(self):
		return unicode(Account.objects.get(id=self.id)).encode("utf-8")

	def __unicode__(self):
		return u'%s [%s]' % (self.__class__.__name__, self.identification())


class EmailAccount(Account):
	class Meta(object):
		app_label = 'requester'
	hostname = models.TextField()
	port = models.IntegerField()
	username = models.EmailField()

	def identification(self):
		return u'%s@%s:%s' % (self.username, self.hostname, self.port)


class ServerAccount(Account):
	class Meta(object):
		app_label = 'requester'
	username = models.TextField()

	def identification(self):
		return u'%s' % (self.username)


class WebServerAccount(ServerAccount):
	class Meta(object):
		app_label = 'requester'
	url = models.TextField()

	def identification(self):
		return u'%s@%s' % (super(WebServerAccount,self).identification(), self.url)


class FTPServerAccount(ServerAccount):
	class Meta(object):
		app_label = 'requester'
	hostname = models.TextField()

	def identification(self):
		return u'%s@%s' % (super(FTPServerAccount,self).identification(), self.hostname)


class Satellite(models.Model):
	class Meta(object):
		app_label = 'requester'
	name = models.TextField()
	identification = models.TextField()
	in_file = models.TextField()
	request_server = models.ForeignKey(WebServerAccount)
	created = models.DateTimeField(auto_now_add=True)
	modified = models.DateTimeField(auto_now=True)

	def __str__(self):
		return unicode(self).encode("utf-8")

	def __unicode__(self):
		return self.name


class Channel(models.Model):
	class Meta(object):
		app_label = 'requester'
	name = models.TextField(db_index=True)
	in_file = models.TextField(db_index=True,null=True)
	satellite = models.ForeignKey(Satellite)
	created = models.DateTimeField(auto_now_add=True)
	modified = models.DateTimeField(auto_now=True)

	def __str__(self):
		return unicode(self).encode("utf-8")

	def __unicode__(self):
		return u'%s (%s)' % (self.name, unicode(self.satellite))


class AutomaticDownload(models.Model):
	class Meta(object):
		app_label = 'requester'
	title = models.TextField(db_index=True)
	area = models.ForeignKey(Area)
	time_range = models.ForeignKey(UTCTimeRange,db_index=True)
	paused = models.BooleanField()
	max_simultaneous_request = models.IntegerField()
	channels = models.ManyToManyField(Channel)
	email_server = models.ForeignKey(EmailAccount)
	root_path = models.TextField()
	created = models.DateTimeField(auto_now_add=True)
	modified = models.DateTimeField(auto_now=True)

	def __str__(self):
		return unicode(self).encode("utf-8")

	def __unicode__(self):
		return self.title

	def pending_requests(self):
		posible_pending = [ r for r in self.request_set.all() if not r.order.downloaded ]
		return [ r for r in posible_pending if r.order.update_downloaded() and not r.order.downloaded ]

	def pending_orders(self):
		return [ r.order for r in self.pending_requests() ]

	def step(self):
		return self.time_range.step()

	def get_next_request_range(self, end_0=None):
		begins = [ r.end for r in self.request_set.all() ]
		begins.append(self.time_range.begin)
		if not end_0 is None:
			begins.append(end_0)
		if total_seconds(self.step()) >= 0:
			f = max
		else:
			f = min
		begin = f(begins)
		end = begin + self.step()
		return begin, end

	def progress(self):
		return str(self.request_set.all().count()) + '/' + str(self.time_range.steps())

	def total_time(self):
		deltas = [ request.total_time() for request in self.request_set.all() ]
		total = None
		for d in deltas:
			total = d if not total else total + d
		return total

	def average_time(self):
		amount = self.request_set.all().count()
		return amount if amount == 0 else (self.total_time() / amount)

	def estimated_time(self):
		return self.average_time() * self.time_range.steps()