import sys
sys.path.append(".")
from django.db import models
from libs.console import show
from datetime import datetime
import factopy
import pytz
import threading

class Stream(factopy.models.Stream):
	class Meta(object):
		app_label = 'plumbing'
	root_path = models.TextField(db_index=True)

	def __unicode__(self):
		return u'%s %s %s' % (unicode(self.pk), self.root_path, unicode(self.tags))

	@classmethod
	def get_stream_from_filename(klass, localfile):
		return "/".join(localfile.split("/")[:-2]) + "/"

	def clone(self):
		s = super(Stream, self).clone()
		s.root_path = self.root_path
		s.save()
		return s