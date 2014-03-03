from django.contrib import admin
from polymorphic.admin import PolymorphicParentModelAdmin, PolymorphicChildModelAdmin, PolymorphicChildModelFilter
from requester.models import Account, FTPServerAccount, WebServerAccount, EmailAccount, Area, UTCTimeRange, GOESRequest, \
	Satellite, Channel, AutomaticDownload
from requester.models.materials import File, Request, Order
from factopy.admin import MaterialAdmin, ProcessAdmin, MaterialChildAdmin, ProcessChildAdmin, ComplexProcessChildAdmin

MaterialAdmin.child_models += (
	(File, MaterialChildAdmin),
	(Request, MaterialChildAdmin),
	(Order, MaterialChildAdmin),
)

class AreaAdmin(admin.ModelAdmin):
	list_display = [ 'name', 'north_latitude', 'south_latitude', 'east_longitude', 'west_longitude']


class UTCTimeRangeAdmin(admin.ModelAdmin):
	list_display = ['begin', 'end']


class SatelliteAdmin(admin.ModelAdmin):
	list_display = [ 'name', 'request_server' ]


class ChannelAdmin(admin.ModelAdmin):
	list_display = [ 'name', 'satellite' ]


class AccountChildAdmin(PolymorphicChildModelAdmin):
	base_model = Account


class AccountAdmin(PolymorphicParentModelAdmin):
	base_model = Account
	list_filter = (PolymorphicChildModelFilter,)
	child_models = (
		(FTPServerAccount, AccountChildAdmin),
		(WebServerAccount, AccountChildAdmin),
		(EmailAccount, AccountChildAdmin),
	)

class RequestAdmin(admin.ModelAdmin):
	list_display = [ 'identification', 'begin', 'end', 'progress', 'downloaded_porcentage', 'total_time' ]

class AutomaticDownloadAdmin(admin.ModelAdmin):
	list_display = [ 'paused','title', 'progress', 'time_range', 'area', 'root_path', 'total_time', 'estimated_time', 'average_time' ]

admin.site.register(Account, AccountAdmin)
admin.site.register(Area, AreaAdmin)
admin.site.register(UTCTimeRange, UTCTimeRangeAdmin)
admin.site.register(GOESRequest, RequestAdmin)
admin.site.register(Satellite, SatelliteAdmin)
admin.site.register(Channel, ChannelAdmin)
admin.site.register(AutomaticDownload, AutomaticDownloadAdmin)
