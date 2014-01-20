from django.contrib import admin
from polymorphic.admin import PolymorphicParentModelAdmin, PolymorphicChildModelAdmin, PolymorphicChildModelFilter
from plumbing.models import TagManager, Stream, MaterialStatus, Material, Process, ProcessOrder, ComplexProcess, Program, File, CollectTimed, CollectChannel, FilterTimed, FilterChannel, FilterSolarElevation, AppendCountToRadiationCoefficient, Compact, SyncImporter
from django.forms import ModelForm


class TagManagerAdmin(admin.ModelAdmin):
	list_display = ['tag_string']


class StreamAdmin(admin.ModelAdmin):
	list_display = ['root_path', 'created', 'modified']


class MaterialChildAdmin(PolymorphicChildModelAdmin):
	base_model = Material


class MaterialAdmin(PolymorphicParentModelAdmin):
	base_model = Material
	list_filter = (PolymorphicChildModelFilter,)
	child_models = (
		(File, MaterialChildAdmin),
	)


class MaterialStatusAdmin(admin.ModelAdmin):
	list_display = ['stream', 'material']


class ProcessChildAdmin(PolymorphicChildModelAdmin):
	base_model = Process


class ProcessInlineForm(ModelForm,object):
	class Meta(object):
			model = ProcessOrder
			fields = ['position','process']


class ProcessOrderInline(admin.TabularInline):
	form = ProcessInlineForm
	model = ProcessOrder
	fk_name = 'complex_process'
	extra = 0 # how many rows to show
	ordering = ["position",]


class ComplexProcessChildAdmin(ProcessChildAdmin):
	list_display = [ 'name', 'description']
	inlines = [ProcessOrderInline]
	search_fields = ['name', 'description', ]


class ProcessAdmin(PolymorphicParentModelAdmin):
	base_model = Process
	list_filter = (PolymorphicChildModelFilter,)
	child_models = (
		(CollectTimed, ProcessChildAdmin),
		(CollectChannel, ProcessChildAdmin),
		(FilterTimed, ProcessChildAdmin),
		(FilterChannel, ProcessChildAdmin),
		(FilterSolarElevation, ProcessChildAdmin),
		(AppendCountToRadiationCoefficient, ProcessChildAdmin),
		(Compact, ProcessChildAdmin),
		(ComplexProcess, ComplexProcessChildAdmin),
		(Program, ComplexProcessChildAdmin),
		(SyncImporter, ProcessChildAdmin),
	)


admin.site.register(TagManager, TagManagerAdmin)
admin.site.register(Stream, StreamAdmin)
admin.site.register(Material, MaterialAdmin)
admin.site.register(MaterialStatus, MaterialStatusAdmin)
admin.site.register(Process, ProcessAdmin)