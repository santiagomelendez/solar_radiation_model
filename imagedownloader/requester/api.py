from requester.models import AutomaticDownload
from tastypie.authentication import SessionAuthentication
from tastypie.resources import ModelResource


class AutomaticDownloadResource(ModelResource):
	class Meta(object):		
		queryset = AutomaticDownload.objects.all()
		resource_name = 'automatic_download'
		filtering = {
			'created': ['exact', 'lt', 'lte', 'gte', 'gt'],
			'modified': ['exact', 'lt', 'lte', 'gte', 'gt'],
		}
		authentication = SessionAuthentication()