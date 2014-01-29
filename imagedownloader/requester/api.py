from plumbing.models import Stream, Material, MaterialStatus, Process
from tastypie import fields
from libs.tastypie_polymorphic import PolymorphicModelResource, ModelResource, SessionAuthentication


class MaterialResource(PolymorphicModelResource):
	class Meta(object):		
		queryset = Material.objects.all()
		resource_name = 'material'
		filtering = {
			'created': ['exact', 'lt', 'lte', 'gte', 'gt'],
			'modified': ['exact', 'lt', 'lte', 'gte', 'gt'],
		}
		authentication = SessionAuthentication()


class MaterialStatusResource(ModelResource):
	material = fields.ForeignKey(MaterialResource, 'material', full=True)
	class Meta(object):
		queryset = MaterialStatus.objects.all()
		resource_name = 'material_status'
		authentication = SessionAuthentication()


class StreamResource(ModelResource):
	class Meta(object):
		queryset = Stream.objects.all()
		resource_name = 'stream'
		filtering = {
			'created': ['exact', 'lt', 'lte', 'gte', 'gt'],
			'modified': ['exact', 'lt', 'lte', 'gte', 'gt'],
		}
	materials = fields.ToManyField(MaterialStatusResource, 'materials', full=True)
	authentication = SessionAuthentication()

	def dehydrate(self, bundle):
		bundle.data['tags'] = bundle.obj.tags.list()
		return bundle


class ProcessResource(PolymorphicModelResource):
	class Meta(object):
		queryset = Process.objects.all()
		resource_name = 'process'
		filtering = {
			'name': ['exact'],
		}
		authentication = SessionAuthentication()

	def dehydrate(self, bundle):
		super(ProcessResource, self).dehydrate(bundle)
		if hasattr(bundle.obj, 'get_ordered_subprocesses'):
			subprocesses = [ {'id': p.id, 'class': p.__class__.__name__, 'name': p.name} for p in bundle.obj.get_ordered_subprocesses() ]
			self.extend_key(bundle.data, self.get_class_name(bundle), { 'processes' : subprocesses })
		return bundle