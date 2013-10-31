from datetime import datetime
import urllib2
from pyquery import PyQuery

def getsat(satellite):
	return satellite.translate(None, "-").lower()

def getmonth(month):
	return datetime.strptime(month.translate(None, "."), "%b %Y")

def getfloat(number):
	number = number.translate(None, " ")
	return 0.0 if number == "" else float(number.split("*")[0])

class CountsShift:
	def coefficient(self, sat):
		return 32.0

class SpaceMeasurement:
	def coefficient(self, sat):
		return 29.0

class PreLaunch:
	def __init__(self):
		table = self.table()
		self.coefficients = self.coefficients(table)
	def table(self):
		page = urllib2.urlopen("http://www.star.nesdis.noaa.gov/smcd/spb/fwu/homepage/GOES_Imager_Vis_PreCal.php")
		pq = PyQuery(page.read())
		return pq("table")[1].findall("tr")
	def coefficients(self, table):
		co = {}
		for r in table[1:]:
			co[getsat(r[0].text_content())] = [ getfloat(e.text_content()) for e in r.getchildren()[1:] ]
		return co
	def coefficient(self, sat):
		return self.coefficients[getsat(sat)]

class PostLaunch:
	def __init__(self):
		table = self.table()
		self.sats = self.satellites(table)
		self.months = self.coefficients(table)
	def table(self):
		page = urllib2.urlopen('http://www.star.nesdis.noaa.gov/smcd/spb/fwu/homepage/GOES_Imager_Vis_OpCal.php')
		pq = PyQuery(page.read())
		return pq("table")[2].findall("tr")
	def satellites(self, table):
		sats = table[1:2]
		sats = [ getsat(s.text_content()) for s in sats[0].getchildren() if not s.text_content() == "" ]
		return sats
	def coefficients(self, table):
		co = {}
		for r in table[2:]:
			size = len(r.getchildren())
			co[getmonth(r[0].text_content())] = [ getfloat(e.text_content()) for e in r.getchildren()[1:] ]
		return co
	def coefficient(self, sat, year, month):
		s = self.sats.index(sat)
		dt = datetime(year, month, 1)
		return self.months[dt][s]

counts_shift = CountsShift()
space_measurement = SpaceMeasurement()
prelaunch = PreLaunch()
postlaunch = PostLaunch()
