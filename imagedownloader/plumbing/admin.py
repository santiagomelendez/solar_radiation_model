from django.contrib import admin
from polymorphic.admin import PolymorphicParentModelAdmin, \
    PolymorphicChildModelAdmin, PolymorphicChildModelFilter
from plumbing.models import FilterTimed, FilterChannel, \
    FilterSolarElevation, AppendCountToRadiationCoefficient, Compact, \
    SyncImporter
from django.forms import ModelForm
from factopy.admin import MaterialAdmin, ProcessAdmin, MaterialChildAdmin, \
    ProcessChildAdmin

ProcessAdmin.child_models += (
		(FilterTimed, ProcessChildAdmin),
		(FilterChannel, ProcessChildAdmin),
		(FilterSolarElevation, ProcessChildAdmin),
		(AppendCountToRadiationCoefficient, ProcessChildAdmin),
		(Compact, ProcessChildAdmin),
		(SyncImporter, ProcessChildAdmin),
	)
