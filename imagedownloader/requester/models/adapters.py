from django.db import models
from factopy.models import Adapt
from core import Area, Channel, EmailAccount
from libs.console import total_seconds
from datetime import datetime, timedelta


class NOAAAdapt(Adapt):
	class Meta(object):
		app_label = 'requester'
	title = models.TextField(db_index=True)
	area = models.ForeignKey('Area')
	paused = models.BooleanField()
	max_simultaneous_request = models.IntegerField()
	channels = models.ManyToManyField('Channel')
	email_server = models.ForeignKey('EmailAccount')
	request_server = models.ForeignKey('WebServerAccount')
	root_path = models.TextField()
	created = models.DateTimeField(auto_now_add=True)
	modified = models.DateTimeField(auto_now=True)
	begin = models.DateTimeField(null=True)
	end = models.DateTimeField(null=True)

	def __str__(self):
		return unicode(self).encode("utf-8")

	def __unicode__(self):
		return self.title

	def step(self):
		begin = datetime.utcnow().replace(tzinfo=pytz.UTC) if self.begin is None else self.begin
		end = datetime.utcnow().replace(tzinfo=pytz.UTC) if self.end is None else self.end
		diff = total_seconds(end - begin)
		return timedelta(days=(diff / abs(diff)))

	def steps(self):
		return int(total_seconds(self.end - self.begin) / total_seconds(self.step()))

	def contains(self, dt):
		timezoned = dt.replace(tzinfo=pytz.UTC)
		return self.begin <= timezoned and self.end >= timezoned

	def pending_requests(self):
		posible_pending = [ r for r in self.request_set.all() if not r.order.downloaded ]
		return [ r for r in posible_pending if r.order.update_downloaded() and not r.order.downloaded ]

	def pending_orders(self):
		return [ r.order for r in self.pending_requests() ]

	def get_next_request_range(self, end_0=None):
		begins = [ r.end for r in self.request_set.all() ]
		begins.append(self.begin)
		if not end_0 is None:
			begins.append(end_0)
		if total_seconds(self.step()) >= 0:
			f = max
		else:
			f = min
		begin = f(begins)
		end = begin + self.step()
		return begin, end

	def progress(self):
		return str(self.request_set.all().count()) + '/' + str(self.steps())

	def total_time(self):
		deltas = [ request.total_time() for request in self.request_set.all() ]
		total = None
		for d in deltas:
			total = d if not total else total + d
		return total

	def average_time(self):
		amount = self.request_set.all().count()
		return amount if amount == 0 else (self.total_time() / amount)

	def estimated_time(self):
		return self.average_time() * self.steps()