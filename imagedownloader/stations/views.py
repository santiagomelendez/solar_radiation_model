from django.shortcuts import render_to_response
from django.template import RequestContext
from django.http import HttpResponseRedirect
from django.shortcuts import render_to_response
from stations.forms import DocumentForm
from stations.models import Configuration

def upload(request):
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            config, end = form.get_configuration()
            config.backup_file(request.FILES['backup'], end)
            return render_to_response('stations/upload.html', {'form': form})
    form = DocumentForm()
    return render_to_response('stations/upload.html', {'form': form}, context_instance=RequestContext(request))
