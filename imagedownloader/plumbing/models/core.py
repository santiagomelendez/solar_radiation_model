import sys
sys.path.append(".")
from django.db import models
from requester.models import *
from decimal import Decimal
from polymodels.models import PolymorphicModel, PolymorphicManager
import glob
from libs.file import netcdf as nc
import calendar


class TagManager(models.Model):
	class Meta:
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


class Stream(models.Model,object):
	class Meta:
		app_label = 'plumbing'
	root_path = models.TextField(db_index=True)
	tags = models.ForeignKey(TagManager, related_name='stream', default=TagManager.empty)
	created = models.DateTimeField(editable=False,default=datetime.utcnow().replace(tzinfo=pytz.UTC))
	modified = models.DateTimeField(default=datetime.utcnow().replace(tzinfo=pytz.UTC))

	def __str__(self):
		return self.root_path

	def save(self, *args, **kwargs):
		""" On save, update timestamps """
		now = datetime.utcnow().replace(tzinfo=pytz.UTC)
		self.tags.save()
		if not self.id:
			self.created = now
		self.modified = now
		return super(Stream, self).save(*args, **kwargs)

	@classmethod
	def get_stream_from_file(klass, localfile):
		return "/".join(localfile.split("/")[:-2]) + "/"

	def get_root_path_files(self):
		return glob.glob(self.root_path + "*/*.nc")

	def sync_files(self):
		files_tmp = []
		for f in self.get_root_path_files():
			f, n = File.objects.get_or_create(localname=str(f))
			if n: f.save()
			fs, n = FileStatus.objects.get_or_create(file=f,stream=self)
			if n: fs.save()
			#self.files.add(fs)
			files_tmp.append([fs, True]) 
		return files_tmp

	def clone(self):
		t = self.tags.clone()
		t.save()
		s = Stream(root_path=self.root_path,tags=t)
		s.save()
		return s

	def unprocessed(self):
		return self.files.filter(processed=False)

	def sorted_files(self):
		return sorted(self.unprocessed(), key=lambda fs: fs.file.filename())

	def empty(self):
		pending = self.unprocessed()
		return len(pending) == 0


class File(models.Model):
	class Meta:
		app_label = 'plumbing'
	localname = models.TextField(unique = True, db_index=True, default="")
	created = models.DateTimeField(auto_now_add=True)
	modified = models.DateTimeField(auto_now=True)

	def filename(self):
		return self.localname

	def __gt__(self, obj):
		return self.datetime() > obj.datetime()

	def __lt__(self, obj):
		return self.datetime() < obj.datetime()

	def __str__(self):
		return self.localname.split("/")[-1]

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
			root, n = nc.open(self.completepath())
			lat = nc.getvar(root,'lat')[:]
			lon = nc.getvar(root, 'lon')[:]
			nc.close(root)
		except RuntimeError:
			show(self.completepath())
			lat, lon = None, None
		return lat, lon

	def completepath(self):
		return os.path.expanduser(os.path.normpath(self.localname))


class FileStatus(models.Model):
	class Meta:
		app_label = 'plumbing'
		unique_together = ("file", "stream")
	file = models.ForeignKey('File', related_name='stream')
	stream = models.ForeignKey(Stream, related_name='files')
	processed = models.BooleanField()

	def clone_for(self, stream):
		fs = FileStatus(file=self.file,stream=stream)
		fs.save()
		return fs


class ProcessManager(PolymorphicManager,object):

	def all(self, *argc, **argv):
		return super(ProcessManager,self).all(*argc, **argv).select_subclasses()

	def filter(self, *argc, **argv):
		return super(ProcessManager,self).filter(*argc, **argv).select_subclasses()


class Process(PolymorphicModel):
	class Meta:
		app_label = 'plumbing'
	objects = ProcessManager()
	name = models.TextField(db_index=True)
	description = models.TextField(db_index=True)

	def __str__(self):
		return self.__class__.__name__ + ' [' + self.name + ']'

	def mark_with_tags(self, stream):
		pass


class ComplexProcess(Process):
	class Meta:
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


class ProcessOrder(models.Model):
	class Meta:
		app_label = 'plumbing'
	position = models.IntegerField()
	process = models.ForeignKey('Process', related_name='used_by')
	complex_process = models.ForeignKey(ComplexProcess)
