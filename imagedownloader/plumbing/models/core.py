import sys
sys.path.append(".")
from django.db import models
from polymorphic import PolymorphicModel, PolymorphicManager
import glob
from libs.file import netcdf as nc
from libs.console import show
import calendar
from datetime import datetime, timedelta
import pytz
import os
import re


class TagManager(models.Model):
	class Meta(object):
		app_label = 'plumbing'
	tag_string = models.TextField(db_index=True, default="")

	@classmethod
	def empty(klass):
		tm = TagManager()
		tm.save()
		return tm

	def exist(self, tag):
		return tag in self.list()

	def list(self):
		l = self.tag_string.split(",")
		if u"" in l: l.remove(u"")
		return l

	def insert_first(self, tag):
		if not self.exist(tag):
			self.tag_string = (tag + "," + self.tag_string)  if len(self.tag_string) > 0 else tag
			self.save()

	def append(self,tag):
		if not self.exist(tag):
			self.tag_string += ("," + tag) if len(self.tag_string) > 0 else tag
			self.save()

	def clone(self):
		t = TagManager(tag_string=self.tag_string)
		t.save()
		return t

	def make_filename(self):
		return ".".join(self.list())

	def __str__(self):
		return unicode(self).encode("utf-8")

	def __unicode__(self):
		return u'[%s]' % self.tag_string


class Stream(models.Model,object):
	class Meta(object):
		app_label = 'plumbing'
	root_path = models.TextField(db_index=True)
	tags = models.ForeignKey(TagManager, related_name='stream', default=TagManager.empty)
	created = models.DateTimeField(editable=False,default=datetime.utcnow().replace(tzinfo=pytz.UTC))
	modified = models.DateTimeField(default=datetime.utcnow().replace(tzinfo=pytz.UTC))

	def __str__(self):
		return unicode(self).encode("utf-8")

	def __unicode__(self):
		return u'%s %s %s' % (unicode(self.pk), self.root_path, unicode(self.tags))

	def save(self, *args, **kwargs):
		""" On save, update timestamps """
		now = datetime.utcnow().replace(tzinfo=pytz.UTC)
		self.tags.save()
		if not self.pk:
			self.created = now
		self.modified = now
		return super(Stream, self).save(*args, **kwargs)

	@classmethod
	def get_stream_from_filename(klass, localfile):
		return "/".join(localfile.split("/")[:-2]) + "/"

	def clone(self):
		t = self.tags.clone()
		t.save()
		s = Stream(root_path=self.root_path,tags=t)
		s.save()
		return s

	def unprocessed(self):
		return self.materials.filter(processed=False)

	def empty(self):
		pending = self.unprocessed()
		return len(pending) == 0


class Material(PolymorphicModel, object):
	class Meta(object):
		app_label = 'plumbing'
	objects = PolymorphicManager()
	created = models.DateTimeField(auto_now_add=True)
	modified = models.DateTimeField(auto_now=True)

	def __str__(self):
		return unicode(Material.objects.get(pk=self.pk)).encode("utf-8")

	def __unicode__(self):
		return u'(created: %s, modified: %s)' % (unicode(self.created), unicode(self.modified))


class MaterialStatus(models.Model):
	class Meta(object):
		app_label = 'plumbing'
		verbose_name_plural = 'Material statuses'
		unique_together = ("material", "stream")
	material = models.ForeignKey('Material', related_name='stream')
	stream = models.ForeignKey(Stream, related_name='materials')
	processed = models.BooleanField()

	def __str__(self):
		return unicode(self).encode("utf-8")

	def __unicode__(self):
		return u'%s -> %s' % (unicode(self.stream), unicode(self.material))

	def clone_for(self, stream):
		cloned_material_status = MaterialStatus(material=self.material,stream=stream)
		cloned_material_status.save()
		return cloned_material_status


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


class Process(PolymorphicModel):
	class Meta(object):
		app_label = 'plumbing'
		verbose_name_plural = 'Processes'
	objects = PolymorphicManager()
	name = models.TextField(db_index=True)
	description = models.TextField(db_index=True)

	def __str__(self):
		return unicode(Process.objects.get(pk=self.pk)).encode("utf-8")

	def __unicode__(self):
		return u'%s [%s]' % (self.__class__.__name__, self.name)

	def mark_with_tags(self, stream):
		pass


class ComplexProcess(Process):
	class Meta(object):
		app_label = 'plumbing'
	processes = models.ManyToManyField('Process', through='ProcessOrder', related_name='complex_process')

	def encapsulate_in_array(self, streams):
		if not streams.__class__ in [list, tuple]:
			streams = [ streams ]
		return streams

	def do(self, stream):
		ps = self.processes.all().order_by('used_by__position')
		for subprocess in ps:
			show("Running " + str(subprocess))
			stream = self.encapsulate_in_array(stream)
			tmp_results = []
			for s in stream:
				if not s.empty():
					result = subprocess.type_cast().do(s)
					subprocess.mark_with_tags(s)
					tmp_results += self.encapsulate_in_array(result)
			stream = tmp_results
		return stream


class Adapter(Process):
	class Meta(object):
		app_label = 'plumbing'
	stream = models.ForeignKey(Stream)

	def update(self):
		raise Exception("Subclass responsability")


class ProcessOrder(models.Model):
	class Meta(object):
		app_label = 'plumbing'
	position = models.IntegerField()
	process = models.ForeignKey('Process', related_name='used_by')
	complex_process = models.ForeignKey(ComplexProcess)

	def __str__(self):
		return unicode(self).encode("utf-8")

	def __unicode__(self):
		return unicode(self.process)