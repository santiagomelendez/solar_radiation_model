from stations.models import OpticFilter, Brand, Product, Device, SensorCalibration, Position, Station, Configuration, Measurement
from tastypie import fields
from tastypie.authentication import SessionAuthentication
from tastypie.resources import ModelResource
from libs.tastypie_polymorphic import PolymorphicModelResource


class BrandResource(ModelResource):
	class Meta(object):
		queryset = Brand.objects.all()
		resource_name = 'brand'
		authentication = SessionAuthentication()


class ProductResource(ModelResource):
	class Meta(object):
		queryset = Product.objects.all()
		resource_name = 'product'
		authentication = SessionAuthentication()


class DeviceResource(PolymorphicModelResource):
	class Meta(object):
		queryset = Device.objects.all()
		resource_name = 'device'
		authentication = SessionAuthentication()


class OpticFilterResource(ModelResource):
	class Meta(object):
		queryset = OpticFilter.objects.all()
		resource_name = 'optic_filter'
		authentication = SessionAuthentication()


class SensorCalibrationResource(ModelResource):
	class Meta(object):
		queryset = SensorCalibration.objects.all()
		resource_name = 'sensor_calibration'
		authentication = SessionAuthentication()


class StationResource(ModelResource):
	materials = fields.ToManyField('PositionResource', 'coordinates', full=True)
	class Meta(object):
		queryset = Station.objects.all()
		resource_name = 'station'
		authentication = SessionAuthentication()


class PositionResource(ModelResource):
	station = fields.ForeignKey(StationResource, 'station', full=True)
	class Meta(object):
		queryset = Position.objects.all()
		resource_name = 'position'
		authentication = SessionAuthentication()


class ConfigurationResource(ModelResource):
	class Meta(object):
		queryset = Configuration.objects.all()
		resource_name = 'configuration'
		authentication = SessionAuthentication()


class MeasurementResource(ModelResource):
	class Meta(object):
		queryset = Measurement.objects.all()
		resource_name = 'measurement'
		authentication = SessionAuthentication()