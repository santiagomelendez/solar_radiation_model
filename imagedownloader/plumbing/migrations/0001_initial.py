# -*- coding: utf-8 -*-
import datetime
from south.db import db
from south.v2 import SchemaMigration
from django.db import models


class Migration(SchemaMigration):

    def forwards(self, orm):
        # Adding model 'TagManager'
        db.create_table('plumbing_tagmanager', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('tag_string', self.gf('django.db.models.fields.TextField')(default='', db_index=True)),
        ))
        db.send_create_signal('plumbing', ['TagManager'])

        # Adding model 'Stream'
        db.create_table('plumbing_stream', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('root_path', self.gf('django.db.models.fields.TextField')(db_index=True)),
            ('tags', self.gf('django.db.models.fields.related.ForeignKey')(related_name='stream', to=orm['plumbing.TagManager'])),
            ('created', self.gf('django.db.models.fields.DateTimeField')(auto_now_add=True, blank=True)),
            ('modified', self.gf('django.db.models.fields.DateTimeField')(auto_now=True, blank=True)),
        ))
        db.send_create_signal('plumbing', ['Stream'])

        # Adding model 'File'
        db.create_table('plumbing_file', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('localname', self.gf('django.db.models.fields.TextField')(default='', unique=True, db_index=True)),
            ('created', self.gf('django.db.models.fields.DateTimeField')(auto_now_add=True, blank=True)),
            ('modified', self.gf('django.db.models.fields.DateTimeField')(auto_now=True, blank=True)),
        ))
        db.send_create_signal('plumbing', ['File'])

        # Adding model 'FileStatus'
        db.create_table('plumbing_filestatus', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('file', self.gf('django.db.models.fields.related.ForeignKey')(related_name='stream', to=orm['plumbing.File'])),
            ('stream', self.gf('django.db.models.fields.related.ForeignKey')(related_name='files', to=orm['plumbing.Stream'])),
            ('processed', self.gf('django.db.models.fields.BooleanField')(default=False)),
        ))
        db.send_create_signal('plumbing', ['FileStatus'])

        # Adding unique constraint on 'FileStatus', fields ['file', 'stream']
        db.create_unique('plumbing_filestatus', ['file_id', 'stream_id'])

        # Adding model 'Process'
        db.create_table('plumbing_process', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('content_type', self.gf('django.db.models.fields.related.ForeignKey')(related_name=u'+', to=orm['contenttypes.ContentType'])),
            ('name', self.gf('django.db.models.fields.TextField')(db_index=True)),
            ('description', self.gf('django.db.models.fields.TextField')(db_index=True)),
        ))
        db.send_create_signal('plumbing', ['Process'])

        # Adding model 'ComplexProcess'
        db.create_table('plumbing_complexprocess', (
            ('process_ptr', self.gf('django.db.models.fields.related.OneToOneField')(to=orm['plumbing.Process'], unique=True, primary_key=True)),
        ))
        db.send_create_signal('plumbing', ['ComplexProcess'])

        # Adding model 'ProcessOrder'
        db.create_table('plumbing_processorder', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('position', self.gf('django.db.models.fields.IntegerField')()),
            ('process', self.gf('django.db.models.fields.related.ForeignKey')(related_name='used_by', to=orm['plumbing.Process'])),
            ('complex_process', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['plumbing.ComplexProcess'])),
        ))
        db.send_create_signal('plumbing', ['ProcessOrder'])

        # Adding model 'FilterSolarElevation'
        db.create_table('plumbing_filtersolarelevation', (
            ('process_ptr', self.gf('django.db.models.fields.related.OneToOneField')(to=orm['plumbing.Process'], unique=True, primary_key=True)),
            ('minimum', self.gf('django.db.models.fields.DecimalField')(max_digits=4, decimal_places=2)),
            ('hourly_longitude', self.gf('django.db.models.fields.DecimalField')(default='0.00', max_digits=5, decimal_places=2)),
        ))
        db.send_create_signal('plumbing', ['FilterSolarElevation'])

        # Adding model 'Collect'
        db.create_table('plumbing_collect', (
            ('process_ptr', self.gf('django.db.models.fields.related.OneToOneField')(to=orm['plumbing.Process'], unique=True, primary_key=True)),
        ))
        db.send_create_signal('plumbing', ['Collect'])

        # Adding model 'CollectTimed'
        db.create_table('plumbing_collecttimed', (
            ('collect_ptr', self.gf('django.db.models.fields.related.OneToOneField')(to=orm['plumbing.Collect'], unique=True, primary_key=True)),
            ('yearly', self.gf('django.db.models.fields.BooleanField')(default=False)),
            ('monthly', self.gf('django.db.models.fields.BooleanField')(default=False)),
            ('weekly', self.gf('django.db.models.fields.BooleanField')(default=False)),
        ))
        db.send_create_signal('plumbing', ['CollectTimed'])

        # Adding model 'CollectChannel'
        db.create_table('plumbing_collectchannel', (
            ('collect_ptr', self.gf('django.db.models.fields.related.OneToOneField')(to=orm['plumbing.Collect'], unique=True, primary_key=True)),
        ))
        db.send_create_signal('plumbing', ['CollectChannel'])

        # Adding model 'FilterChannel'
        db.create_table('plumbing_filterchannel', (
            ('process_ptr', self.gf('django.db.models.fields.related.OneToOneField')(to=orm['plumbing.Process'], unique=True, primary_key=True)),
        ))
        db.send_create_signal('plumbing', ['FilterChannel'])

        # Adding M2M table for field channels on 'FilterChannel'
        db.create_table('plumbing_filterchannel_channels', (
            ('id', models.AutoField(verbose_name='ID', primary_key=True, auto_created=True)),
            ('filterchannel', models.ForeignKey(orm['plumbing.filterchannel'], null=False)),
            ('channel', models.ForeignKey(orm['requester.channel'], null=False))
        ))
        db.create_unique('plumbing_filterchannel_channels', ['filterchannel_id', 'channel_id'])

        # Adding model 'FilterTimed'
        db.create_table('plumbing_filtertimed', (
            ('process_ptr', self.gf('django.db.models.fields.related.OneToOneField')(to=orm['plumbing.Process'], unique=True, primary_key=True)),
        ))
        db.send_create_signal('plumbing', ['FilterTimed'])

        # Adding M2M table for field time_range on 'FilterTimed'
        db.create_table('plumbing_filtertimed_time_range', (
            ('id', models.AutoField(verbose_name='ID', primary_key=True, auto_created=True)),
            ('filtertimed', models.ForeignKey(orm['plumbing.filtertimed'], null=False)),
            ('utctimerange', models.ForeignKey(orm['requester.utctimerange'], null=False))
        ))
        db.create_unique('plumbing_filtertimed_time_range', ['filtertimed_id', 'utctimerange_id'])

        # Adding model 'AppendCountToRadiationCoefficient'
        db.create_table('plumbing_appendcounttoradiationcoefficient', (
            ('process_ptr', self.gf('django.db.models.fields.related.OneToOneField')(to=orm['plumbing.Process'], unique=True, primary_key=True)),
        ))
        db.send_create_signal('plumbing', ['AppendCountToRadiationCoefficient'])

        # Adding model 'Compact'
        db.create_table('plumbing_compact', (
            ('process_ptr', self.gf('django.db.models.fields.related.OneToOneField')(to=orm['plumbing.Process'], unique=True, primary_key=True)),
            ('extension', self.gf('django.db.models.fields.TextField')()),
            ('resultant_stream', self.gf('django.db.models.fields.related.ForeignKey')(default=None, to=orm['plumbing.Stream'], null=True)),
        ))
        db.send_create_signal('plumbing', ['Compact'])

        # Adding model 'Image'
        db.create_table('plumbing_image', (
            ('file_ptr', self.gf('django.db.models.fields.related.OneToOneField')(to=orm['plumbing.File'], unique=True, primary_key=True)),
        ))
        db.send_create_signal('plumbing', ['Image'])

        # Adding model 'Program'
        db.create_table('plumbing_program', (
            ('complexprocess_ptr', self.gf('django.db.models.fields.related.OneToOneField')(to=orm['plumbing.ComplexProcess'], unique=True, primary_key=True)),
            ('stream', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['plumbing.Stream'])),
        ))
        db.send_create_signal('plumbing', ['Program'])


    def backwards(self, orm):
        # Removing unique constraint on 'FileStatus', fields ['file', 'stream']
        db.delete_unique('plumbing_filestatus', ['file_id', 'stream_id'])

        # Deleting model 'TagManager'
        db.delete_table('plumbing_tagmanager')

        # Deleting model 'Stream'
        db.delete_table('plumbing_stream')

        # Deleting model 'File'
        db.delete_table('plumbing_file')

        # Deleting model 'FileStatus'
        db.delete_table('plumbing_filestatus')

        # Deleting model 'Process'
        db.delete_table('plumbing_process')

        # Deleting model 'ComplexProcess'
        db.delete_table('plumbing_complexprocess')

        # Deleting model 'ProcessOrder'
        db.delete_table('plumbing_processorder')

        # Deleting model 'FilterSolarElevation'
        db.delete_table('plumbing_filtersolarelevation')

        # Deleting model 'Collect'
        db.delete_table('plumbing_collect')

        # Deleting model 'CollectTimed'
        db.delete_table('plumbing_collecttimed')

        # Deleting model 'CollectChannel'
        db.delete_table('plumbing_collectchannel')

        # Deleting model 'FilterChannel'
        db.delete_table('plumbing_filterchannel')

        # Removing M2M table for field channels on 'FilterChannel'
        db.delete_table('plumbing_filterchannel_channels')

        # Deleting model 'FilterTimed'
        db.delete_table('plumbing_filtertimed')

        # Removing M2M table for field time_range on 'FilterTimed'
        db.delete_table('plumbing_filtertimed_time_range')

        # Deleting model 'AppendCountToRadiationCoefficient'
        db.delete_table('plumbing_appendcounttoradiationcoefficient')

        # Deleting model 'Compact'
        db.delete_table('plumbing_compact')

        # Deleting model 'Image'
        db.delete_table('plumbing_image')

        # Deleting model 'Program'
        db.delete_table('plumbing_program')


    models = {
        'contenttypes.contenttype': {
            'Meta': {'ordering': "('name',)", 'unique_together': "(('app_label', 'model'),)", 'object_name': 'ContentType', 'db_table': "'django_content_type'"},
            'app_label': ('django.db.models.fields.CharField', [], {'max_length': '100'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'model': ('django.db.models.fields.CharField', [], {'max_length': '100'}),
            'name': ('django.db.models.fields.CharField', [], {'max_length': '100'})
        },
        'plumbing.appendcounttoradiationcoefficient': {
            'Meta': {'object_name': 'AppendCountToRadiationCoefficient', '_ormbases': ['plumbing.Process']},
            'process_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['plumbing.Process']", 'unique': 'True', 'primary_key': 'True'})
        },
        'plumbing.collect': {
            'Meta': {'object_name': 'Collect', '_ormbases': ['plumbing.Process']},
            'process_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['plumbing.Process']", 'unique': 'True', 'primary_key': 'True'})
        },
        'plumbing.collectchannel': {
            'Meta': {'object_name': 'CollectChannel', '_ormbases': ['plumbing.Collect']},
            'collect_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['plumbing.Collect']", 'unique': 'True', 'primary_key': 'True'})
        },
        'plumbing.collecttimed': {
            'Meta': {'object_name': 'CollectTimed', '_ormbases': ['plumbing.Collect']},
            'collect_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['plumbing.Collect']", 'unique': 'True', 'primary_key': 'True'}),
            'monthly': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'weekly': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'yearly': ('django.db.models.fields.BooleanField', [], {'default': 'False'})
        },
        'plumbing.compact': {
            'Meta': {'object_name': 'Compact', '_ormbases': ['plumbing.Process']},
            'extension': ('django.db.models.fields.TextField', [], {}),
            'process_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['plumbing.Process']", 'unique': 'True', 'primary_key': 'True'}),
            'resultant_stream': ('django.db.models.fields.related.ForeignKey', [], {'default': 'None', 'to': "orm['plumbing.Stream']", 'null': 'True'})
        },
        'plumbing.complexprocess': {
            'Meta': {'object_name': 'ComplexProcess', '_ormbases': ['plumbing.Process']},
            'process_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['plumbing.Process']", 'unique': 'True', 'primary_key': 'True'}),
            'processes': ('django.db.models.fields.related.ManyToManyField', [], {'related_name': "'complex_process'", 'symmetrical': 'False', 'through': "orm['plumbing.ProcessOrder']", 'to': "orm['plumbing.Process']"})
        },
        'plumbing.file': {
            'Meta': {'object_name': 'File'},
            'created': ('django.db.models.fields.DateTimeField', [], {'auto_now_add': 'True', 'blank': 'True'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'localname': ('django.db.models.fields.TextField', [], {'default': "''", 'unique': 'True', 'db_index': 'True'}),
            'modified': ('django.db.models.fields.DateTimeField', [], {'auto_now': 'True', 'blank': 'True'})
        },
        'plumbing.filestatus': {
            'Meta': {'unique_together': "(('file', 'stream'),)", 'object_name': 'FileStatus'},
            'file': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'stream'", 'to': "orm['plumbing.File']"}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'processed': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'stream': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'files'", 'to': "orm['plumbing.Stream']"})
        },
        'plumbing.filterchannel': {
            'Meta': {'object_name': 'FilterChannel', '_ormbases': ['plumbing.Process']},
            'channels': ('django.db.models.fields.related.ManyToManyField', [], {'to': "orm['requester.Channel']", 'db_index': 'True', 'symmetrical': 'False'}),
            'process_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['plumbing.Process']", 'unique': 'True', 'primary_key': 'True'})
        },
        'plumbing.filtersolarelevation': {
            'Meta': {'object_name': 'FilterSolarElevation', '_ormbases': ['plumbing.Process']},
            'hourly_longitude': ('django.db.models.fields.DecimalField', [], {'default': "'0.00'", 'max_digits': '5', 'decimal_places': '2'}),
            'minimum': ('django.db.models.fields.DecimalField', [], {'max_digits': '4', 'decimal_places': '2'}),
            'process_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['plumbing.Process']", 'unique': 'True', 'primary_key': 'True'})
        },
        'plumbing.filtertimed': {
            'Meta': {'object_name': 'FilterTimed', '_ormbases': ['plumbing.Process']},
            'process_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['plumbing.Process']", 'unique': 'True', 'primary_key': 'True'}),
            'time_range': ('django.db.models.fields.related.ManyToManyField', [], {'to': "orm['requester.UTCTimeRange']", 'db_index': 'True', 'symmetrical': 'False'})
        },
        'plumbing.image': {
            'Meta': {'object_name': 'Image', '_ormbases': ['plumbing.File']},
            'file_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['plumbing.File']", 'unique': 'True', 'primary_key': 'True'})
        },
        'plumbing.process': {
            'Meta': {'object_name': 'Process'},
            'content_type': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "u'+'", 'to': "orm['contenttypes.ContentType']"}),
            'description': ('django.db.models.fields.TextField', [], {'db_index': 'True'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'name': ('django.db.models.fields.TextField', [], {'db_index': 'True'})
        },
        'plumbing.processorder': {
            'Meta': {'object_name': 'ProcessOrder'},
            'complex_process': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['plumbing.ComplexProcess']"}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'position': ('django.db.models.fields.IntegerField', [], {}),
            'process': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'used_by'", 'to': "orm['plumbing.Process']"})
        },
        'plumbing.program': {
            'Meta': {'object_name': 'Program', '_ormbases': ['plumbing.ComplexProcess']},
            'complexprocess_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['plumbing.ComplexProcess']", 'unique': 'True', 'primary_key': 'True'}),
            'stream': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['plumbing.Stream']"})
        },
        'plumbing.stream': {
            'Meta': {'object_name': 'Stream'},
            'created': ('django.db.models.fields.DateTimeField', [], {'auto_now_add': 'True', 'blank': 'True'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'modified': ('django.db.models.fields.DateTimeField', [], {'auto_now': 'True', 'blank': 'True'}),
            'root_path': ('django.db.models.fields.TextField', [], {'db_index': 'True'}),
            'tags': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'stream'", 'to': "orm['plumbing.TagManager']"})
        },
        'plumbing.tagmanager': {
            'Meta': {'object_name': 'TagManager'},
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'tag_string': ('django.db.models.fields.TextField', [], {'default': "''", 'db_index': 'True'})
        },
        'requester.account': {
            'Meta': {'object_name': 'Account'},
            'created': ('django.db.models.fields.DateTimeField', [], {'auto_now_add': 'True', 'blank': 'True'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'modified': ('django.db.models.fields.DateTimeField', [], {'auto_now': 'True', 'blank': 'True'}),
            'password': ('django.db.models.fields.TextField', [], {})
        },
        'requester.channel': {
            'Meta': {'object_name': 'Channel'},
            'created': ('django.db.models.fields.DateTimeField', [], {'auto_now_add': 'True', 'blank': 'True'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'in_file': ('django.db.models.fields.TextField', [], {'null': 'True', 'db_index': 'True'}),
            'modified': ('django.db.models.fields.DateTimeField', [], {'auto_now': 'True', 'blank': 'True'}),
            'name': ('django.db.models.fields.TextField', [], {'db_index': 'True'}),
            'satellite': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['requester.Satellite']"})
        },
        'requester.satellite': {
            'Meta': {'object_name': 'Satellite'},
            'created': ('django.db.models.fields.DateTimeField', [], {'auto_now_add': 'True', 'blank': 'True'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'identification': ('django.db.models.fields.TextField', [], {}),
            'in_file': ('django.db.models.fields.TextField', [], {}),
            'modified': ('django.db.models.fields.DateTimeField', [], {'auto_now': 'True', 'blank': 'True'}),
            'name': ('django.db.models.fields.TextField', [], {}),
            'request_server': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['requester.WebServerAccount']"})
        },
        'requester.serveraccount': {
            'Meta': {'object_name': 'ServerAccount', '_ormbases': ['requester.Account']},
            'account_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['requester.Account']", 'unique': 'True', 'primary_key': 'True'}),
            'username': ('django.db.models.fields.TextField', [], {})
        },
        'requester.utctimerange': {
            'Meta': {'object_name': 'UTCTimeRange'},
            'begin': ('django.db.models.fields.DateTimeField', [], {}),
            'created': ('django.db.models.fields.DateTimeField', [], {'auto_now_add': 'True', 'blank': 'True'}),
            'end': ('django.db.models.fields.DateTimeField', [], {}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'modified': ('django.db.models.fields.DateTimeField', [], {'auto_now': 'True', 'blank': 'True'})
        },
        'requester.webserveraccount': {
            'Meta': {'object_name': 'WebServerAccount', '_ormbases': ['requester.ServerAccount']},
            'serveraccount_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['requester.ServerAccount']", 'unique': 'True', 'primary_key': 'True'}),
            'url': ('django.db.models.fields.TextField', [], {})
        }
    }

    complete_apps = ['plumbing']