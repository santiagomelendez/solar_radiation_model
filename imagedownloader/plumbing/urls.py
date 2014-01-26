from django.conf.urls import patterns, include, url
from tastypie.api import Api
from plumbing.api import StreamResource, MaterialResource, MaterialStatusResource, ProcessResource


api_v1 = Api(api_name='v1')
api_v1.register(StreamResource())
api_v1.register(MaterialResource())
api_v1.register(MaterialStatusResource())
api_v1.register(ProcessResource())

urlpatterns = patterns('',
	url(r'^api/', include(api_v1.urls)),
	url(r'^program/$', 'plumbing.views.index'),
	url(r'^program/execute/(?P<program_id>\d+)/$', 'plumbing.views.execute'),
	url(r'^program/status$', 'plumbing.views.status'),
	url(r'^program/update$', 'plumbing.views.update'),
)