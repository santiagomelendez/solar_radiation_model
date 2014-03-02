from django.conf.urls import patterns, include, url

urlpatterns = patterns('',
	url(r'^program/$', 'plumbing.views.index'),
	url(r'^program/execute/(?P<program_id>\d+)/$', 'plumbing.views.execute'),
	url(r'^program/status$', 'plumbing.views.status'),
	url(r'^program/update$', 'plumbing.views.update'),
)