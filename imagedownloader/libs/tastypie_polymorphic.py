from tastypie.resources import ModelResource
from tastypie.serializers import Serializer
from tastypie.authentication import SessionAuthentication


class PolymorphicModelResource(ModelResource):
	def get_class_name(self, bundle):
		return unicode(bundle.obj.__class__.__name__).lower()
	def extend_key(self, dictionary, key, subdictionary):
		if not key in dictionary: dictionary[key] = {}
		dictionary[key] = dict(dictionary[key].items() + subdictionary.items())
	def deserialize(self, obj):
		return dict((attr, obj.__dict__[attr]) for attr in obj.__dict__ if not (attr[-3:] == '_id' or attr in ['_state']))
	def add_foreign_keys(self, bundle, key, foreign_keys):
		foreign_keys.remove('polymorphic_ctype_id')
		ids = [ fk for fk in foreign_keys if not fk[-len('_ptr_id'):] == '_ptr_id' ]
		extras = dict((k[:-3],self.deserialize(getattr(bundle.obj,k[:-3]))) for k in ids)
		print extras
		self.extend_key(bundle.data, self.get_class_name(bundle), extras)
	def dehydrate(self, bundle):
		bundle = super(PolymorphicModelResource,self).dehydrate(bundle)
		foreign_keys = [ k for k in bundle.obj.__dict__.keys() if k[-len('_id'):] == '_id' ]
		keys = [ unicode(k) for k in bundle.data.keys() + [ '_state', 'polymorphic_ctype_id' ] + foreign_keys ]
		extras = dict((k,v) for k,v in bundle.obj.__dict__.iteritems() if not unicode(k) in keys )
		self.extend_key(bundle.data, self.get_class_name(bundle), extras)
		self.add_foreign_keys(bundle, self.get_class_name(bundle), foreign_keys)
		return bundle