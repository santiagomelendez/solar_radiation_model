from django.http import HttpResponse
from django.template import Context, loader
from django.shortcuts import render, redirect
from plumbing.models import Program, Importer

# Create your views here.

def index(request):
	program_list = Program.objects.all()
	template = loader.get_template('plumbing/index.html')
	context = Context({
		'program_list': program_list,
	})
	return HttpResponse(template.render(context))

def execute(request, program_id):
	programs = Program.objects.filter(id=program_id)
	programs[0].execute()
	return redirect('/plumbing/status')

def status(request):
	programs = Program.objects.all()
	return render(request, 'plumbing/status.html', {'programs': programs})

def update(request):
	new_importers = Importer.setup_unloaded()
	template = loader.get_template('plumbing/update.html')
	context = Context({
		'new_importers': new_importers,
	})
	return HttpResponse(template.render(context))
