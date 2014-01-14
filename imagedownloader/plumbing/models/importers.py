import sys
sys.path.append(".")
from core import File, ComplexProcess


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
		root = Dataset(self.completepath(),'r')
		lat = root.variables['lat'][:]
		lon = root.variables['lon'][:]
		root.close()
		return lat, lon

	def completepath(self):
		return os.path.expanduser(os.path.normpath(self.localname))


class Program(ComplexProcess):
	class Meta(object):
			app_label = 'plumbing'
	stream = models.ForeignKey(Stream)

	def downloaded_files(self):
		return [ f for request in self.automatic_download.request_set.all() for f in request.order.file_set.filter(downloaded=True).order_by('localname') ]

	def source(self):
		files = self.downloaded_files()
		files.sort()
		return { 'all': files }

	def execute(self):
		self.do(self.stream)