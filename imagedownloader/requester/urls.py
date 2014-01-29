from django.conf.urls import patterns, include, url
from tastypie.api import Api
from plumbing.api import *


api_v1 = Api(api_name='v1')
api_v1.register(StreamResource())
api_v1.register(MaterialResource())
api_v1.register(MaterialStatusResource())
api_v1.register(ProcessResource())


urlpatterns = patterns('',
	url(r'^index$', 'plumbing.views.index'),
	url(r'^execute/(?P<program_id>\d+)/$', 'plumbing.views.execute'),
	url(r'^status$', 'plumbing.views.status'),
	url(r'^update$', 'plumbing.views.update'),
	url(r'^api/', include(api_v1.urls)),
)