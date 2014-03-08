from django.db import models
from factopy.models import Material
from processes import FTPServerAccount
from datetime import datetime, timedelta
import pytz  # 3rd party
import os
from django.db.models import Min, Max
from libs.console import total_seconds
import re
import calendar
from adapters import NOAAAdapt


class Request(Material):
	class Meta(object):
		app_label = 'requester'
	adapt = models.ForeignKey('NOAAAdapt',db_index=True)
	begin = models.DateTimeField(db_index=True)
	end = models.DateTimeField(db_index=True)
	aged = models.BooleanField(default=False)

	def __str__(self):
		return unicode(self).encode("utf-8")

	def __unicode__(self):
		return u'%s->%s (%s)' % (unicode(self.begin), unicode(self.end), self.order.identification)

	def get_channels(self):
		return self.adapt.channels.all()

	def get_satellite(self):
		return self.get_channels()[0].satellite

	def get_request_server(self):
		# For each request one server is used (the same satellite is shared by all the channels)
		return self.adapt().request_server

	def get_area(self):
		return self.adapt.area

	def get_email_server(self):
		return self.adapt.email_server

	def get_timerange(self):
		return (self.begin, self.end)

	def get_order(self):
		orders = [] #Order.objects.filter(request=self)
		if len(orders) == 0:
			self.save()
			order = {} # Order()
			order.request = self
			order.identification = ''
			order.save()
		else:
			order = orders[0]
		return order

	def identification(self):
		return self.get_order().identification

	def progress(self):
		return self.order.progress()

	def downloaded_porcentage(self):
		return self.order.downloaded_porcentage()

	def total_time(self):
		return self.order.total_time()


"""class Order(Material):
	class Meta(object):
		app_label = 'requester'
	request = models.OneToOneField('Request',db_index=True)
	server = models.ForeignKey('FTPServerAccount', null=True)
	identification = models.TextField(db_index=True)
	downloaded = models.BooleanField()
	empty_flag = models.BooleanField()

	def __str__(self):
		return unicode(self).encode("utf-8")

	def __unicode__(self):
		return self.identification

	def pending_files(self):
		return self.file_set.filter(downloaded=False)

	def update_downloaded(self):
		self.downloaded = True if self.empty_flag else (len(self.file_set.all()) > 0 and len(self.pending_files()) == 0)
		self.save()
		return True

	def progress(self):
		return str(self.file_set.filter(downloaded=True).count())+'/'+str(self.file_set.all().count())

	def downloaded_porcentage(self):
		files = self.file_set.all().count()
		ratio = 0. if files == 0 else (self.file_set.filter(downloaded=True).count() / float(self.file_set.all().count()))
		ratio = 1. if self.empty_flag else ratio
		return '%.2f' % (ratio * 100.)

	def total_time(self):
		result = self.file_set.filter(downloaded=True).aggregate(Min('begin_download'), Max('end_download'))
		if result['begin_download__min'] is None:
			result['begin_download__min'] = datetime.utcnow().replace(tzinfo=pytz.UTC)
		if result['end_download__max'] is None:
			result['end_download__max'] = datetime.utcnow().replace(tzinfo=pytz.UTC)
		return result['end_download__max'] - result['begin_download__min']

	def average_speed(self):
		if not self.downloaded:
			speeds = [ float(f.download_speed()) for f in self.file_set.filter(begin_download__isnull = False)]
			avg = (sum(speeds) / len(speeds)) if len(speeds) > 0 else 0
		else:
			avg = 0
		return '%.2f' % (avg * 8)

	def download_speed(self):
		if not self.downloaded:
			speeds = [ float(f.download_speed()) for f in self.file_set.filter(begin_download__isnull = False, end_download__isnull = True)]
			avg = (sum(speeds) / len(speeds)) if len(speeds) > 0 else 0
		else:
			avg = 0
		return '%.2f' % (avg * 8)

	def year(self):
		return self.request.get_timerange()[0].year

	def julian_day(self):
		return min(self.request.get_timerange()[0].timetuple()[7],self.request.get_timerange()[1].timetuple()[7])"""


class File(Material):
	class Meta(object):
		app_label = 'requester'
	localname = models.TextField(unique = True, db_index=True, default="")
	#order = models.ForeignKey('Order', db_index=True, null=True)
	remotename = models.TextField(null=True)
	size = models.IntegerField(null=True)
	downloaded = models.BooleanField(db_index=True)
	begin_download = models.DateTimeField(null=True,db_index=True)
	end_download = models.DateTimeField(null=True,db_index=True)
	failures = models.IntegerField(default=0)

	def __str__(self):
		return unicode(self).encode("utf-8")

	def __unicode__(self):
		return self.filename()

	def __gt__(self, obj):
		return self.datetime() > obj.datetime()

	def __lt__(self, obj):
		return self.datetime() < obj.datetime()

	def filename(self):
		return self.localname.split('/')[-1]

	def completepath(self):
		return os.path.expanduser(os.path.normpath(self.localname))

	def localsize(self):
		path = self.completepath()
		return os.stat(path).st_size if os.path.isfile(path) else 0

	def downloaded_porcentage(self):
		return '%.2f' % (0 if self.size == 0 else ((float(self.localsize()) / self.size) * 100.))

	def progress(self):
		return str(self.localsize()) + '/' + str(self.size)

	def download_speed(self):
		now = datetime.utcnow().replace(tzinfo=pytz.UTC)
		begin = now if self.begin_download is None else self.begin_download
		last = now if self.end_download is None else self.end_download
		speed_in_bytes = self.localsize() / total_seconds(last - begin) if last != begin else 0
		return '%.2f' % (speed_in_bytes / 1024.)

	def verify_copy_status(self):
		if self.size is None:
			self.size = 0
		if self.localsize() != self.size or self.size == 0 or self.localsize() == 0:
			self.downloaded = False
			self.end_download = None
			self.begin_download = None
			self.save()
			self.order.downloaded = False
			self.order.save()

	def channel(self):
		res = re.search('BAND_([0-9]*)\.', self.completepath())
		return str(res.groups(0)[0]) if res else None

	def satellite(self):
		res = re.search('([a-z]*[0-9]*)\.', self.filename().replace("pkg.", ""))
		return str(res.groups(0)[0]) if res else None

	def deduce_year_fraction(self, year, rest):
		if rest[0][0] == "M":
			month = int(rest[0][1:])
			return datetime(year,month,calendar.monthrange(year,month)[1])
		elif rest[0][0] == "W":
			return datetime(year, 1 , 1) +  timedelta(weeks=int(rest[0][1:]), days=-1)
		else:
			days = int(rest[0])
			time = rest[1]
			date = datetime(year, 1, 1) + timedelta(days - 1)
			return date.replace(hour=int(time[0:2]), minute=int(time[2:4]), second=int(time[4:6]))

	def datetime(self):
		t_info = self.filename().replace("pkg.", "").split(".")
		year = int(t_info[1])
		return self.deduce_year_fraction(year, t_info[2:])

	def latlon(self):
		try:
			root = nc.open(self.completepath())[0]
			lat = nc.getvar(root,'lat')[:]
			lon = nc.getvar(root, 'lon')[:]
			nc.close(root)
		except RuntimeError:
			show(self.completepath())
			lat, lon = None, None
		return lat, lon


class Image(File):
	class Meta(object):
		app_label = 'requester'

	def satellite(self):
		sats = Satellite.objects.find(in_file = self.satellite())
		return sats[0] if len(sats) > 0 else None

	def channel(self):
		sat = self.image_satellite()
		chs = sat.channel_set.find(in_file=self.channel())
		return chs[0] if len(chs) > 0 else None

	def latlon(self):
		if self.channel() is None:
			return None, None
		root = nc.open(self.completepath())[0]
		lat = nc.getvar(root, "lat")[:]
		lon = nc.getvar(root, "lon")[:]
		nc.close(root)
		return lat, lon

	def completepath(self):
		return os.path.expanduser(os.path.normpath(self.localname))