from django.conf.urls import patterns, include, url
from tastypie.api import Api
from stations.api import BrandResource, ProductResource, DeviceResource, OpticFilterResource, SensorCalibrationResource, StationResource, PositionResource, ConfigurationResource, MeasurementResource


api_v1 = Api(api_name='v1')
api_v1.register(BrandResource())
api_v1.register(ProductResource())
api_v1.register(DeviceResource())
api_v1.register(OpticFilterResource())
api_v1.register(SensorCalibrationResource())
api_v1.register(StationResource())
api_v1.register(PositionResource())
api_v1.register(ConfigurationResource())
api_v1.register(MeasurementResource())

urlpatterns = patterns('',
	url(r'^api/', include(api_v1.urls)),
	url(r'^stations/upload', 'stations.views.upload'),
)