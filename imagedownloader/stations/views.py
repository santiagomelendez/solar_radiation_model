from django.shortcuts import render_to_response
from django.template import RequestContext
from stations.forms import DocumentForm

def upload(request):
	if request.method == 'POST':
		form = DocumentForm(request.POST, request.FILES)
		if form.is_valid():
			config, end, rows, between, refresh_presision = form.get_configuration(request)
			if config: config.register_measurements(end, rows, between, refresh_presision)
			return render_to_response('stations/upload.html', {'form': form})
	form = DocumentForm()
	return render_to_response('stations/upload.html', {'form': form}, context_instance=RequestContext(request))