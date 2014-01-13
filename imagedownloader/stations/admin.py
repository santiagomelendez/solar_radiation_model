from django.contrib import admin
from stations.models import *
from django.forms import ModelForm


class BrandAdmin(admin.ModelAdmin):
	list_display = [ 'name', ]


class OpticFilterAdmin(admin.ModelAdmin):
	list_display = [ 'name', ]


class ProductAdmin(admin.ModelAdmin):
	list_display = [ 'brand', 'name' ]
	list_display_links = list_display[1:]


class DeviceAdmin(admin.ModelAdmin):
	list_display = [ 'product', 'serial_number', 'description' ]
	list_display_links = list_display[1:]


class SensorAdmin(DeviceAdmin):
	list_display = [ 'product', 'serial_number', 'description', 'optic_filter' ]
	list_display_links = list_display[1:]


class InclinedSupportAdmin(DeviceAdmin):
	list_display = [ 'product', 'serial_number', 'description', 'angle', ]
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
admin.site.register(Sensor, SensorAdmin)
admin.site.register(Datalogger, DeviceAdmin)
admin.site.register(ShadowBall, DeviceAdmin)
admin.site.register(Tracker, DeviceAdmin)
admin.site.register(InclinedSupport, InclinedSupportAdmin)
admin.site.register(SensorCalibration, SensorCalibrationAdmin)
admin.site.register(Configuration, ConfigurationAdmin)
admin.site.register(Measurement, MeasurementAdmin)
