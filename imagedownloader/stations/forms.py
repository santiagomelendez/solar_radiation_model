# In forms.py...
from django import forms
from stations.models import Configuration
from datetime import datetime
import pytz

class DocumentForm(forms.Form):
    def __init__(self, *args, **kwargs):
        super(DocumentForm, self).__init__(*args, **kwargs)
        self.fields["configuration"] = forms.ChoiceField(label="Configuration", choices = [(c.id, c) for c in Configuration.actives()])
        self.fields["end"] = forms.DateTimeField(label="End configuration", initial=datetime.utcnow())
        self.fields["backup"] = forms.FileField(label='Select the file with measurements', help_text='max. 42 megabytes')
        self.fields["between"] = forms.IntegerField(label='Between')
        self.fields["refresh_presision"] = forms.IntegerField(label='Refresh presision')
    def get_configuration(self):
        label = self.cleaned_data['configuration']
        end = self.cleaned_data["end"].replace(tzinfo=pytz.UTC)
        between = self.cleaned_data["between"]
        refresh_presision = self.cleaned_data["refresh_presision"]
        return None if not label else dict(self.fields['configuration'].choices)[int(label)], end, between, refresh_presision

