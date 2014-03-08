from django.contrib import admin
from polymorphic.admin import PolymorphicParentModelAdmin, PolymorphicChildModelAdmin, PolymorphicChildModelFilter
from requester.models import Area, Satellite, Channel
from requester.models import NOAAAdapt
from requester.models import Account, FTPServerAccount, WebServerAccount, EmailAccount, GOESRequest, NOAAEMailChecker, NOAAFTPDownloader
from requester.models.materials import File, Request #, Order
from factopy.admin import MaterialAdmin, ProcessAdmin, MaterialChildAdmin, ProcessChildAdmin, ComplexProcessChildAdmin

MaterialAdmin.child_models += (
	(File, MaterialChildAdmin),
	(Request, MaterialChildAdmin),
	#(Order, MaterialChildAdmin),
)

ProcessAdmin.child_models += (
	(NOAAAdapt, ProcessChildAdmin),
	(GOESRequest, ProcessChildAdmin),
	(NOAAEMailChecker, ProcessChildAdmin),
	(NOAAFTPDownloader, ProcessChildAdmin),
)

class AreaAdmin(admin.ModelAdmin):
	list_display = [ 'name', 'north_latitude', 'south_latitude', 'east_longitude', 'west_longitude']


class SatelliteAdmin(admin.ModelAdmin):
	list_display = [ 'name' ]


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


admin.site.register(Account, AccountAdmin)
admin.site.register(Area, AreaAdmin)
admin.site.register(Satellite, SatelliteAdmin)
admin.site.register(Channel, ChannelAdmin)
