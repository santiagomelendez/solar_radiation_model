# -*- coding: utf-8 -*-
import datetime
from south.db import db
from south.v2 import SchemaMigration
from django.db import models


class Migration(SchemaMigration):

    def forwards(self, orm):
        # Adding model 'FileStatus'
        db.create_table('plumbing_filestatus', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('file', self.gf('django.db.models.fields.related.ForeignKey')(related_name='used_by', to=orm['plumbing.File'])),
            ('stream', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['plumbing.Stream'])),
            ('processed', self.gf('django.db.models.fields.BooleanField')(default=False)),
        ))
        db.send_create_signal('plumbing', ['FileStatus'])

        # Removing M2M table for field files on 'Stream'
        db.delete_table('plumbing_stream_files')


    def backwards(self, orm):
        # Deleting model 'FileStatus'
        db.delete_table('plumbing_filestatus')

        # Adding M2M table for field files on 'Stream'
        db.create_table('plumbing_stream_files', (
            ('id', models.AutoField(verbose_name='ID', primary_key=True, auto_created=True)),
            ('stream', models.ForeignKey(orm['plumbing.stream'], null=False)),
            ('file', models.ForeignKey(orm['plumbing.file'], null=False))
        ))
        db.create_unique('plumbing_stream_files', ['stream_id', 'file_id'])


    models = {
        'plumbing.collectchannel': {
            'Meta': {'object_name': 'CollectChannel', '_ormbases': ['plumbing.Process']},
            'process_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['plumbing.Process']", 'unique': 'True', 'primary_key': 'True'})
        },
        'plumbing.collecttimed': {
            'Meta': {'object_name': 'CollectTimed', '_ormbases': ['plumbing.Process']},
            'monthly': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'process_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['plumbing.Process']", 'unique': 'True', 'primary_key': 'True'})
        },
        'plumbing.compact': {
            'Meta': {'object_name': 'Compact', '_ormbases': ['plumbing.Process']},
            'extension': ('django.db.models.fields.TextField', [], {}),
            'process_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['plumbing.Process']", 'unique': 'True', 'primary_key': 'True'})
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
            'Meta': {'object_name': 'FileStatus'},
            'file': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'used_by'", 'to': "orm['plumbing.File']"}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'processed': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'stream': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['plumbing.Stream']"})
        },
        'plumbing.filterchannel': {
            'Meta': {'object_name': 'FilterChannel', '_ormbases': ['plumbing.Process']},
            'channels': ('django.db.models.fields.related.ManyToManyField', [], {'to': "orm['requester.Channel']", 'db_index': 'True', 'symmetrical': 'False'}),
            'process_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['plumbing.Process']", 'unique': 'True', 'primary_key': 'True'})
        },
        'plumbing.filtersolarelevation': {
            'Meta': {'object_name': 'FilterSolarElevation', '_ormbases': ['plumbing.Process']},
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
            'description': ('django.db.models.fields.TextField', [], {'db_index': 'True'}),
            'executing': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'name': ('django.db.models.fields.TextField', [], {'db_index': 'True'}),
            'progress': ('django.db.models.fields.DecimalField', [], {'max_digits': '5', 'decimal_places': '2'})
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
            'files': ('django.db.models.fields.related.ManyToManyField', [], {'related_name': "'streams'", 'symmetrical': 'False', 'through': "orm['plumbing.FileStatus']", 'to': "orm['plumbing.File']"}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'modified': ('django.db.models.fields.DateTimeField', [], {'auto_now': 'True', 'blank': 'True'}),
            'root_path': ('django.db.models.fields.TextField', [], {'unique': 'True', 'db_index': 'True'})
        },
        'plumbing.transformcountstoradiation': {
            'Meta': {'object_name': 'TransformCountsToRadiation', '_ormbases': ['plumbing.Process']},
            'calibrated_coefficient': ('django.db.models.fields.DecimalField', [], {'max_digits': '5', 'decimal_places': '3'}),
            'counts_shift': ('django.db.models.fields.IntegerField', [], {}),
            'process_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['plumbing.Process']", 'unique': 'True', 'primary_key': 'True'}),
            'space_measurement': ('django.db.models.fields.DecimalField', [], {'max_digits': '5', 'decimal_places': '3'})
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