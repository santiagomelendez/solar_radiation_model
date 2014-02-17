import sys
sys.path.append(".")
from django.db import models
from polymorphic import PolymorphicModel, PolymorphicManager
from libs.console import show
from datetime import datetime
import pytz
import threading


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
		return unicode(Material.objects.get(id=self.id)).encode("utf-8")

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


class Process(PolymorphicModel,object):
	class Meta(object):
		app_label = 'plumbing'
		verbose_name_plural = 'Processes'
	objects = PolymorphicManager()
	name = models.TextField(db_index=True)
	description = models.TextField(db_index=True)

	def __str__(self):
		return unicode(Process.objects.get(id=self.id)).encode("utf-8")

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

	def get_ordered_subprocesses(self):
		return self.processes.all().order_by('used_by__position')

	def do(self, stream):
		ps = self.get_ordered_subprocesses()
		for subprocess in ps:
			show("Running " + str(subprocess))
			stream = self.encapsulate_in_array(stream)
			tmp_results = []
			for s in stream:
				if not s.empty():
					result = subprocess.do(s)
					subprocess.mark_with_tags(s)
					tmp_results += self.encapsulate_in_array(result)
			stream = tmp_results
		return stream


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


class Collect(Process):
	class Meta(object):
		app_label = 'plumbing'

	def get_key(self, material_status):
		raise AttributeError("'Collect' object has no attribute 'get_key'")

	def get_keys(self, stream):
		return set([ self.get_key(fs) for fs in stream.materials.all() ])

	def init_empty_streams(self, stream):
		keys = self.get_keys(stream)
		resultant_stream = {}
		for k in keys:
			resultant_stream[k] = stream.clone()
		return resultant_stream

	def do(self, stream):
		resultant_stream = self.init_empty_streams(stream)
		for fs in stream.materials.all():
			fs.clone_for(resultant_stream[self.get_key(fs)])
			fs.processed=True
			fs.save()
		for k in resultant_stream.keys():
			resultant_stream[k].tags.append(k)
		return resultant_stream.values()

	def mark_with_tags(self, stream):
		# Don't used because these process always return multiple streams
		pass


class Filter(Process):
	class Meta(object):
		app_label = 'plumbing'

	def should_be_cloned(self, material_status):
		return False

	def do(self, stream):
		resultant_stream = stream.clone()
		for fs in stream.materials.all():
			if self.should_be_cloned(fs):
				fs.clone_for(resultant_stream)
			fs.processed=True
			fs.save()
		return resultant_stream

	def mark_with_tags(self, stream):
		# Don't used because these process is transparent in the name
		pass


class Adapter(Process):
	class Meta(object):
		app_label = 'plumbing'
	stream = models.ForeignKey(Stream, null=True, default=None)

	def update(self):
		raise Exception("Subclass responsability")


class Importer(Adapter):
	class Meta(object):
			app_label = 'plumbing'
	frequency = models.IntegerField(default=15*60) # It is expressed in seconds

	@classmethod
	def setup_unloaded(klass):
		importers = [ i for i in klass.objects.all() if not hasattr(i,"thread") ]
		for i in importers:
			i.thread = threading.Timer(i.frequency, i.update)
			i.thread.start()
		return importers
