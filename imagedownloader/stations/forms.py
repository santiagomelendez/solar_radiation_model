# In forms.py...
from django import forms
from stations.models import Configuration
from datetime import datetime
import os
import pytz
from importer import from_csv, from_xls


class DocumentForm(forms.Form,object):

	def __init__(self, *args, **kwargs):
		super(DocumentForm, self).__init__(*args, **kwargs)
		self.fields["configuration"] = forms.ChoiceField(label="Configuration", choices = [(c.id, c) for c in Configuration.actives()])
		self.fields["end"] = forms.DateTimeField(label="End configuration", initial=datetime.utcnow().replace(tzinfo=pytz.UTC))
		self.fields["backup"] = forms.FileField(label='Select the file with measurements', help_text='max. 42 megabytes')
		self.fields["between"] = forms.IntegerField(label='Between')
		self.fields["refresh_presision"] = forms.IntegerField(label='Refresh presision')

	def get_configuration(self, request):
		label = self.cleaned_data['configuration']
		configuration = dict(self.fields['configuration'].choices)[int(label)]
		end = self.cleaned_data["end"].replace(tzinfo=pytz.UTC)
		between = self.cleaned_data["between"]
		refresh_presision = self.cleaned_data["refresh_presision"]
		rows, backup_name = self.process_rows(request, configuration)
		return None if not label or not backup_name or len(rows) is 0 else configuration, end, rows, between, refresh_presision

	def process_rows(self,request,configuration):
		f = request.FILES['backup']
		backup_name = configuration.get_backup_filename(f.name)
		with open(backup_name, 'wb') as destination:
			for chunk in f.chunks():
				destination.write(chunk)
		try:
			return getattr(self, "t_%s_to_rows" % backup_name.split(".")[-1])(backup_name), backup_name
		except Exception:
			os.remove(backup_name)
			return [], None

	def t_csv_to_rows(self, filename):
		utc_diff = -3
		timestamp_col = 0
		channel = 1
		skip_rows = 3
		return from_csv(filename, utc_diff, timestamp_col, channel, skip_rows)

	def t_xls_to_rows(self, filename):
		utc_diff = -3
		i_sheet = 1
		x_year = 1
		x_julian = 2
		x_timestamp = 3
		x_value = 9
		y_from = 10
		return from_xls(filename, utc_diff, i_sheet, x_year, x_julian, x_timestamp, x_value, y_from)