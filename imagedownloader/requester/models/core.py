from django.db import models
from datetime import datetime
import pytz
from decimal import Decimal
from polymorphic import PolymorphicModel, PolymorphicManager

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


class Satellite(models.Model):
	class Meta(object):
		app_label = 'requester'
	name = models.TextField()
	identification = models.TextField()
	in_file = models.TextField()
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
	satellite = models.ForeignKey('Satellite')
	created = models.DateTimeField(auto_now_add=True)
	modified = models.DateTimeField(auto_now=True)

	def __str__(self):
		return unicode(self).encode("utf-8")

	def __unicode__(self):
		return u'%s (%s)' % (self.name, unicode(self.satellite))


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
