from django.contrib import admin
from plumbing.models import *
from django.forms import ModelForm


class TagManagerAdmin(admin.ModelAdmin):
	list_display = ['tag_string']


class StreamAdmin(admin.ModelAdmin):
	list_display = ['root_path', 'created', 'modified']


class FileAdmin(admin.ModelAdmin):
	list_display = ['localname', 'created', 'modified']


class FileStatusAdmin(admin.ModelAdmin):
	list_display = ['stream', 'file']


class ProcessInlineForm(ModelForm,object):
	class Meta(object):
			model = ProcessOrder
			fields = ['position','process']

	def __init__(self, *args, **kwargs):
		super(ProcessInlineForm, self).__init__(*args, **kwargs)
		self.fields['process'].queryset = Process.objects.all() 


class ProcessOrderInline(admin.TabularInline):
	form = ProcessInlineForm
	model = ProcessOrder
	fk_name = 'complex_process'
	extra = 0 # how many rows to show
	ordering = ["position",]


class ComplexProcessAdmin(admin.ModelAdmin):
	list_display = [ 'name', 'description']
	inlines = [ProcessOrderInline]
	search_fields = ['name', 'description', ]


class ProgramAdmin(admin.ModelAdmin):
	list_display = [ 'name', 'description', 'stream']
	inlines = [ProcessOrderInline]
	search_fields = ['name', 'description', ]


class CompactAdmin(admin.ModelAdmin):
	list_display= [ 'name', 'description', 'extension']


class CollectTimedAdmin(admin.ModelAdmin):
	list_display = [ 'name', 'description']


class CollectChannelAdmin(admin.ModelAdmin):
	list_display = [ 'name', 'description' ]


class FilterAdmin(admin.ModelAdmin):
	list_display = [ 'name', 'description']


class FilterSolarElevationAdmin(admin.ModelAdmin):
	list_display = [ 'name', 'description', 'minimum']


class AppendCountToRadiationCoefficientAdmin(admin.ModelAdmin):
	list_display = [ 'name', 'description']


admin.site.register(TagManager, TagManagerAdmin)
admin.site.register(Stream, StreamAdmin)
admin.site.register(File, FileAdmin)
admin.site.register(FileStatus, FileStatusAdmin)
admin.site.register(AppendCountToRadiationCoefficient, AppendCountToRadiationCoefficientAdmin)
admin.site.register(ComplexProcess, ComplexProcessAdmin)
admin.site.register(Compact, CompactAdmin)
admin.site.register(Program, ProgramAdmin)
admin.site.register(FilterTimed, FilterAdmin)
admin.site.register(FilterChannel, FilterAdmin)
admin.site.register(CollectTimed, CollectTimedAdmin)
admin.site.register(CollectChannel, CollectChannelAdmin)
admin.site.register(FilterSolarElevation, FilterSolarElevationAdmin)
