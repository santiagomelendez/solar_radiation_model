from django.contrib import admin
from polymorphic.admin import PolymorphicParentModelAdmin, PolymorphicChildModelAdmin, PolymorphicChildModelFilter
from stations.models import Position, Station, Brand, Product, OpticFilter, SensorCalibration, Configuration, Measurement, Device, Sensor, Datalogger, ShadowBall, Tracker, InclinedSupport


class BrandAdmin(admin.ModelAdmin):
	list_display = [ 'name', ]


class OpticFilterAdmin(admin.ModelAdmin):
	list_display = [ 'name', ]


class ProductAdmin(admin.ModelAdmin):
	list_display = [ 'brand', 'name' ]
	list_display_links = list_display[1:]


class DeviceChildAdmin(PolymorphicChildModelAdmin):
	base_model = Device


class DeviceAdmin(PolymorphicParentModelAdmin):
	base_model = Device
	list_filter = (PolymorphicChildModelFilter,)
	child_models = (
		(Sensor, DeviceChildAdmin),
		(Datalogger, DeviceChildAdmin),
		(Tracker, DeviceChildAdmin),
		(ShadowBall, DeviceChildAdmin),
		(InclinedSupport, DeviceChildAdmin)
	)


class SensorAdmin(DeviceAdmin):
	list_display = [ 'product', 'serial_number', 'description', 'optic_filter' ]
	list_display_links = list_display[1:]


class SensorCalibrationAdmin(admin.ModelAdmin):
	list_display = [ 'coefficient', 'shift' ]
	list_display_links = list_display[:]


class ConfigurationAdmin(admin.ModelAdmin):
	list_display = [ 'position', 'calibration', 'created', 'modified' ]
	list_display_links = list_display[1:]


class MeasurementAdmin(admin.ModelAdmin):
	list_display = [ 'configuration', 'finish', 'mean', 'between', 'refresh_presision' ]
	list_display_links = list_display[:3]


class StationAdmin(admin.ModelAdmin):
	list_display = [ 'name', 'coordinates' ]


class PositionAdmin(admin.ModelAdmin):
	list_display = [ 'station', 'latitude', 'longitude' ]


admin.site.register(Position, PositionAdmin)
admin.site.register(Station, StationAdmin)
admin.site.register(Brand, BrandAdmin)
admin.site.register(Product, ProductAdmin)
admin.site.register(OpticFilter, OpticFilterAdmin)
admin.site.register(Device, DeviceAdmin)
admin.site.register(SensorCalibration, SensorCalibrationAdmin)
admin.site.register(Configuration, ConfigurationAdmin)
admin.site.register(Measurement, MeasurementAdmin)
