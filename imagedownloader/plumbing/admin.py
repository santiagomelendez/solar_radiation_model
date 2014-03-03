from django.contrib import admin
from polymorphic.admin import PolymorphicParentModelAdmin, PolymorphicChildModelAdmin, PolymorphicChildModelFilter
from plumbing.models import Program, CollectTimed, CollectChannel, FilterTimed, FilterChannel, FilterSolarElevation, AppendCountToRadiationCoefficient, Compact, SyncImporter
from django.forms import ModelForm
from factopy.admin import MaterialAdmin, ProcessAdmin, MaterialChildAdmin, ProcessChildAdmin, ComplexProcessChildAdmin

ProcessAdmin.child_models += (
		(CollectTimed, ProcessChildAdmin),
		(CollectChannel, ProcessChildAdmin),
		(FilterTimed, ProcessChildAdmin),
		(FilterChannel, ProcessChildAdmin),
		(FilterSolarElevation, ProcessChildAdmin),
		(AppendCountToRadiationCoefficient, ProcessChildAdmin),
		(Compact, ProcessChildAdmin),
		(Program, ComplexProcessChildAdmin),
		(SyncImporter, ProcessChildAdmin),
	)