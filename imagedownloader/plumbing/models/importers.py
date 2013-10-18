from django.db import models
import glob

class Stream(models.Model):
	class Meta:
		app_label = 'plumbing'
	root_path = models.TextField(unique=True, db_index=True)
	created = models.DateTimeField(auto_now_add=True)
	modified = models.DateTimeField(auto_now=True)
	def __str__(self):
		return self.root_path
	@classmethod
	def get_stream_from_image(cls, localfile):
		return "/".join(localfile.split("/")[:-2]) + "/"
	def get_all_images(self):
		return glob.glob(self.root_path + "*/*.nc")
	def images(self):
		return [ Image.objects.get_or_create(localname=str(f), stream=self) for f in self.get_all_images() ]

class Image(models.Model):
	class Meta:
		app_label = 'plumbing'
	stream = models.ForeignKey(Stream, db_index=True, null=True)
	localname = models.TextField(unique = True, db_index=True)
	created = models.DateTimeField(auto_now_add=True)
	modified = models.DateTimeField(auto_now=True)
	def filename(self):
		return self.localname
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
		root = self.stream.root_path if self.stream else '.'
		return os.path.expanduser(os.path.normpath(root + '/' + self.localname))
	def localsize(self):
		path = self.completepath()
		return os.stat(path).st_size if os.path.isfile(path) else 0
	def __gt__(self, obj):
		return self.datetime() > obj.datetime()
	def __str__(self):
		return self.localname[len(self.stream.root_path):]
