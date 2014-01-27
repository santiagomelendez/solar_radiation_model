from django.db import models
from core import Material
import os
import re
import calendar
from datetime import datetime, timedelta
from libs.file import netcdf as nc
from libs.console import show


class File(Material):
	class Meta(object):
		app_label = 'plumbing'
	localname = models.TextField(unique = True, db_index=True, default="")

	def filename(self):
		return unicode(self.localname)

	def __gt__(self, obj):
		return self.datetime() > obj.datetime()

	def __lt__(self, obj):
		return self.datetime() < obj.datetime()

	def __str__(self):
		return unicode(self.localname).split("/")[-1]

	def localsize(self):
		path = self.completepath()
		return os.stat(path).st_size if os.path.isfile(path) else 0

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

	def completepath(self):
		return os.path.expanduser(os.path.normpath(self.localname))

	def __unicode__(self):
		return self.localname


class Image(File):
	class Meta(object):
		app_label = 'plumbing'

	def channel(self):
		res = re.search('BAND_([0-9]*)\.', self.completepath())
		return str(res.groups(0)[0]) if res else None

	def satellite(self):
		res = self.filename().split(".")
		return str(res[0])

	def datetime(self):
		t_info = self.filename().split(".")
		year = int(t_info[1])
		days = int(t_info[2])
		time = t_info[3]
		date = datetime(year, 1, 1) + timedelta(days - 1)
		return date.replace(hour=int(time[0:2]), minute=int(time[2:4]), second=int(time[4:6]))

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