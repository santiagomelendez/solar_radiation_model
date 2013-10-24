#from django.db import models
from requester.models import *
from model_utils.managers import *
from decimal import Decimal

from django.db import models
import glob

class TagManager(models.Model):
	class Meta:
		app_label = 'plumbing'
	tag_string = models.TextField(db_index=True, default="")
	def append(self,tag):
		self.tag_string += ("," + tag) if len(self.tag_string) > 0 else tag
	def clone(self):
		t = TagManager(tag_string=self.tag_string)
		t.save()
		return t
	def make_filename(self):
		return self.tag_string.split(",").join(".")

class Stream(models.Model):
	class Meta:
		app_label = 'plumbing'
	root_path = models.TextField(db_index=True)
	tags = models.ForeignKey(TagManager, related_name='stream', default=TagManager())
	#files = models.ManyToManyField('FileStatus', through='FileStatus', related_name='streams')
	#name = models.File
	created = models.DateTimeField(auto_now_add=True)
	modified = models.DateTimeField(auto_now=True)
	def __str__(self):
		return self.root_path
	@classmethod
	def get_stream_from_file(cls, localfile):
		return "/".join(localfile.split("/")[:-2]) + "/"
	def get_all_files(self):
		return glob.glob(self.root_path + "*/*.nc")
	def updated_files(self):
		files_tmp = []
		for f in self.get_all_files():
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
	def __str__(self):
		return self.localname.split("/")[-1]
	def localsize(self):
		path = self.completepath()
		return os.stat(path).st_size if os.path.isfile(path) else 0
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
		root = Dataset(self.completepath(),'r')
		lat = root.variables['lat'][:]
		lon = root.variables['lon'][:]
		root.close()
		return lat, lon
	def completepath(self):
		return os.path.expanduser(os.path.normpath(self.localname))

class FileStatus(models.Model):
	class Meta:
        	app_label = 'plumbing'
	file = models.ForeignKey('File', related_name='stream')
	stream = models.ForeignKey(Stream, related_name='files')
	processed = models.BooleanField()
	def clone_for(self, stream):
		fs = FileStatus(file=self.file,stream=stream)
		fs.save()
		return fs

class ProcessManager(InheritanceManager):
	def all(self, *argc, **argv):
		return super(ProcessManager,self).all(*argc, **argv).select_subclasses()
	def filter(self, *argc, **argv):
		return super(ProcessManager,self).filter(*argc, **argv).select_subclasses()

class Process(models.Model):
	class Meta:
        	app_label = 'plumbing'
	objects = ProcessManager()
	name = models.TextField(db_index=True)
	description = models.TextField(db_index=True)
	progress = models.DecimalField(max_digits=5,decimal_places=2)
	executing = models.BooleanField()
	def __str__(self):
		return self.__class__.__name__ + ' [' + self.name + ']'

class ComplexProcess(Process):
	class Meta:
        	app_label = 'plumbing'
	processes = models.ManyToManyField('Process', through='ProcessOrder', related_name='complex_process')
	def encapsulate_in_array(self, stream_or_streams):
		if not stream_or_streams.__class__ in [list, tuple]:
			stream_or_streams = [ stream_or_streams ]
		return stream_or_streams
	def do(self, stream):
		self.progress = 0
		self.executing = True
		self.save()
		ps = self.processes.all().order_by('used_by__position')
		count = 0.0
		for subprocess in ps:
			stream = self.encapsulate_in_array(stream)
			tmp_results = []
			for s in stream:
				result = subprocess.do(s)
				subprocess.mark_with_tags(s)
				tmp_results += self.encapsulate_in_array(result)
			stream = tmp_results
			count += 1
			self.progress = (count / len(ps)) * 100
			self.save()
		self.executing = False
		self.save()
		return stream

class ProcessOrder(models.Model):
	class Meta:
        	app_label = 'plumbing'
	position = models.IntegerField()
	process = models.ForeignKey('Process', related_name='used_by')
	complex_process = models.ForeignKey(ComplexProcess)
