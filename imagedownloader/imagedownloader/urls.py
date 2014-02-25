from django.conf.urls import patterns, include, url
from django.contrib import admin
import factopy
import plumbing, stations, requester

# Api provide an easy way of automatically determining the URL conf.

admin.autodiscover()

urlpatterns = patterns('',
	# Examples:
	url(r'^factopy/', include('factopy.urls')),
	url(r'^plumbing/', include('plumbing.urls')),
	url(r'^stations/', include('stations.urls')),
	url(r'^requester/', include('requester.urls')),

	# Uncomment the admin/doc line below to enable admin documentation:
	url(r'^admin/doc/', include('django.contrib.admindocs.urls')),

	# Uncomment the next line to enable the admin:
	url(r'^admin/', include(admin.site.urls)),
)
