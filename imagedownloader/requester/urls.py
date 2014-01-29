from django.conf.urls import patterns, include, url
from tastypie.api import Api
from requester.api import AutomaticDownloadResource


api_v1 = Api(api_name='v1')
api_v1.register(AutomaticDownloadResource())


urlpatterns = patterns('',
	url(r'^api/', include(api_v1.urls)),
)