# -*- coding: utf-8 -*-
import datetime
from south.db import db
from south.v2 import SchemaMigration
from django.db import models


class Migration(SchemaMigration):

    def forwards(self, orm):
        # Adding model 'Image'
        db.create_table(u'plumbing_image', (
            (u'file_ptr', self.gf('django.db.models.fields.related.OneToOneField')(to=orm['requester.File'], unique=True, primary_key=True)),
        ))
        db.send_create_signal('plumbing', ['Image'])

        # Adding model 'CollectTimed'
        db.create_table(u'plumbing_collecttimed', (
            (u'collect_ptr', self.gf('django.db.models.fields.related.OneToOneField')(to=orm['factopy.Collect'], unique=True, primary_key=True)),
            ('yearly', self.gf('django.db.models.fields.BooleanField')(default=False)),
            ('monthly', self.gf('django.db.models.fields.BooleanField')(default=False)),
            ('weekly', self.gf('django.db.models.fields.BooleanField')(default=False)),
            ('week_day', self.gf('django.db.models.fields.BooleanField')(default=False)),
            ('daily', self.gf('django.db.models.fields.BooleanField')(default=False)),
            ('hourly', self.gf('django.db.models.fields.BooleanField')(default=False)),
            ('slotly', self.gf('django.db.models.fields.BooleanField')(default=False)),
            ('slots_by_day', self.gf('django.db.models.fields.IntegerField')(default=1)),
        ))
        db.send_create_signal('plumbing', ['CollectTimed'])

        # Adding model 'CollectChannel'
        db.create_table(u'plumbing_collectchannel', (
            (u'collect_ptr', self.gf('django.db.models.fields.related.OneToOneField')(to=orm['factopy.Collect'], unique=True, primary_key=True)),
        ))
        db.send_create_signal('plumbing', ['CollectChannel'])

        # Adding model 'FilterChannel'
        db.create_table(u'plumbing_filterchannel', (
            (u'filter_ptr', self.gf('django.db.models.fields.related.OneToOneField')(to=orm['factopy.Filter'], unique=True, primary_key=True)),
        ))
        db.send_create_signal('plumbing', ['FilterChannel'])

        # Adding M2M table for field channels on 'FilterChannel'
        db.create_table(u'plumbing_filterchannel_channels', (
            ('id', models.AutoField(verbose_name='ID', primary_key=True, auto_created=True)),
            ('filterchannel', models.ForeignKey(orm['plumbing.filterchannel'], null=False)),
            ('channel', models.ForeignKey(orm['requester.channel'], null=False))
        ))
        db.create_unique(u'plumbing_filterchannel_channels', ['filterchannel_id', 'channel_id'])

        # Adding model 'FilterTimed'
        db.create_table(u'plumbing_filtertimed', (
            (u'filter_ptr', self.gf('django.db.models.fields.related.OneToOneField')(to=orm['factopy.Filter'], unique=True, primary_key=True)),
            ('yearly', self.gf('django.db.models.fields.BooleanField')(default=False)),
            ('monthly', self.gf('django.db.models.fields.BooleanField')(default=False)),
            ('weekly', self.gf('django.db.models.fields.BooleanField')(default=False)),
            ('week_day', self.gf('django.db.models.fields.BooleanField')(default=False)),
            ('daily', self.gf('django.db.models.fields.BooleanField')(default=False)),
            ('hourly', self.gf('django.db.models.fields.BooleanField')(default=False)),
            ('slotly', self.gf('django.db.models.fields.BooleanField')(default=False)),
            ('slots_by_day', self.gf('django.db.models.fields.IntegerField')(default=1)),
            ('number', self.gf('django.db.models.fields.IntegerField')(default=0)),
            ('error', self.gf('django.db.models.fields.IntegerField')(default=0)),
        ))
        db.send_create_signal('plumbing', ['FilterTimed'])

        # Adding model 'FilterSolarElevation'
        db.create_table(u'plumbing_filtersolarelevation', (
            (u'filter_ptr', self.gf('django.db.models.fields.related.OneToOneField')(to=orm['factopy.Filter'], unique=True, primary_key=True)),
            ('minimum', self.gf('django.db.models.fields.DecimalField')(max_digits=4, decimal_places=2)),
            ('hourly_longitude', self.gf('django.db.models.fields.DecimalField')(default='0.00', max_digits=5, decimal_places=2)),
        ))
        db.send_create_signal('plumbing', ['FilterSolarElevation'])

        # Adding model 'AppendCountToRadiationCoefficient'
        db.create_table(u'plumbing_appendcounttoradiationcoefficient', (
            (u'process_ptr', self.gf('django.db.models.fields.related.OneToOneField')(to=orm['factopy.Process'], unique=True, primary_key=True)),
        ))
        db.send_create_signal('plumbing', ['AppendCountToRadiationCoefficient'])

        # Adding model 'SyncImporter'
        db.create_table(u'plumbing_syncimporter', (
            (u'importer_ptr', self.gf('django.db.models.fields.related.OneToOneField')(to=orm['factopy.Importer'], unique=True, primary_key=True)),
        ))
        db.send_create_signal('plumbing', ['SyncImporter'])

        # Adding model 'Program'
        db.create_table(u'plumbing_program', (
            (u'complexprocess_ptr', self.gf('django.db.models.fields.related.OneToOneField')(to=orm['factopy.ComplexProcess'], unique=True, primary_key=True)),
            ('stream', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['factopy.Stream'])),
        ))
        db.send_create_signal('plumbing', ['Program'])

        # Adding model 'Compact'
        db.create_table(u'plumbing_compact', (
            (u'adapt_ptr', self.gf('django.db.models.fields.related.OneToOneField')(to=orm['factopy.Adapt'], unique=True, primary_key=True)),
            ('extension', self.gf('django.db.models.fields.TextField')()),
        ))
        db.send_create_signal('plumbing', ['Compact'])


    def backwards(self, orm):
        # Deleting model 'Image'
        db.delete_table(u'plumbing_image')

        # Deleting model 'CollectTimed'
        db.delete_table(u'plumbing_collecttimed')

        # Deleting model 'CollectChannel'
        db.delete_table(u'plumbing_collectchannel')

        # Deleting model 'FilterChannel'
        db.delete_table(u'plumbing_filterchannel')

        # Removing M2M table for field channels on 'FilterChannel'
        db.delete_table('plumbing_filterchannel_channels')

        # Deleting model 'FilterTimed'
        db.delete_table(u'plumbing_filtertimed')

        # Deleting model 'FilterSolarElevation'
        db.delete_table(u'plumbing_filtersolarelevation')

        # Deleting model 'AppendCountToRadiationCoefficient'
        db.delete_table(u'plumbing_appendcounttoradiationcoefficient')

        # Deleting model 'SyncImporter'
        db.delete_table(u'plumbing_syncimporter')

        # Deleting model 'Program'
        db.delete_table(u'plumbing_program')

        # Deleting model 'Compact'
        db.delete_table(u'plumbing_compact')


    models = {
        u'contenttypes.contenttype': {
            'Meta': {'ordering': "('name',)", 'unique_together': "(('app_label', 'model'),)", 'object_name': 'ContentType', 'db_table': "'django_content_type'"},
            'app_label': ('django.db.models.fields.CharField', [], {'max_length': '100'}),
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'model': ('django.db.models.fields.CharField', [], {'max_length': '100'}),
            'name': ('django.db.models.fields.CharField', [], {'max_length': '100'})
        },
        'factopy.adapt': {
            'Meta': {'object_name': 'Adapt', '_ormbases': ['factopy.Process']},
            u'process_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['factopy.Process']", 'unique': 'True', 'primary_key': 'True'}),
            'stream': ('django.db.models.fields.related.ForeignKey', [], {'default': 'None', 'to': "orm['factopy.Stream']", 'null': 'True'})
        },
        'factopy.collect': {
            'Meta': {'object_name': 'Collect', '_ormbases': ['factopy.Process']},
            u'process_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['factopy.Process']", 'unique': 'True', 'primary_key': 'True'})
        },
        'factopy.complexprocess': {
            'Meta': {'object_name': 'ComplexProcess', '_ormbases': ['factopy.Process']},
            u'process_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['factopy.Process']", 'unique': 'True', 'primary_key': 'True'}),
            'processes': ('django.db.models.fields.related.ManyToManyField', [], {'related_name': "'complex_process'", 'symmetrical': 'False', 'through': "orm['factopy.ProcessOrder']", 'to': "orm['factopy.Process']"})
        },
        'factopy.filter': {
            'Meta': {'object_name': 'Filter', '_ormbases': ['factopy.Process']},
            u'process_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['factopy.Process']", 'unique': 'True', 'primary_key': 'True'})
        },
        'factopy.importer': {
            'Meta': {'object_name': 'Importer', '_ormbases': ['factopy.Adapt']},
            u'adapt_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['factopy.Adapt']", 'unique': 'True', 'primary_key': 'True'}),
            'frequency': ('django.db.models.fields.IntegerField', [], {'default': '900'})
        },
        'factopy.material': {
            'Meta': {'object_name': 'Material'},
            'created': ('django.db.models.fields.DateTimeField', [], {'auto_now_add': 'True', 'blank': 'True'}),
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'modified': ('django.db.models.fields.DateTimeField', [], {'auto_now': 'True', 'blank': 'True'}),
            'polymorphic_ctype': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'polymorphic_factopy.material_set'", 'null': 'True', 'to': u"orm['contenttypes.ContentType']"})
        },
        'factopy.process': {
            'Meta': {'object_name': 'Process'},
            'description': ('django.db.models.fields.TextField', [], {'db_index': 'True'}),
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'name': ('django.db.models.fields.TextField', [], {'db_index': 'True'}),
            'polymorphic_ctype': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'polymorphic_factopy.process_set'", 'null': 'True', 'to': u"orm['contenttypes.ContentType']"})
        },
        'factopy.processorder': {
            'Meta': {'object_name': 'ProcessOrder'},
            'complex_process': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['factopy.ComplexProcess']"}),
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'position': ('django.db.models.fields.IntegerField', [], {}),
            'process': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'used_by'", 'to': "orm['factopy.Process']"})
        },
        'factopy.stream': {
            'Meta': {'object_name': 'Stream'},
            'created': ('django.db.models.fields.DateTimeField', [], {'default': 'datetime.datetime(2014, 3, 3, 0, 0)'}),
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'modified': ('django.db.models.fields.DateTimeField', [], {'default': 'datetime.datetime(2014, 3, 3, 0, 0)'}),
            'tags': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'stream'", 'to': "orm['factopy.TagManager']"})
        },
        'factopy.tagmanager': {
            'Meta': {'object_name': 'TagManager'},
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'tag_string': ('django.db.models.fields.TextField', [], {'default': "''", 'db_index': 'True'})
        },
        'plumbing.appendcounttoradiationcoefficient': {
            'Meta': {'object_name': 'AppendCountToRadiationCoefficient', '_ormbases': ['factopy.Process']},
            u'process_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['factopy.Process']", 'unique': 'True', 'primary_key': 'True'})
        },
        'plumbing.collectchannel': {
            'Meta': {'object_name': 'CollectChannel', '_ormbases': ['factopy.Collect']},
            u'collect_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['factopy.Collect']", 'unique': 'True', 'primary_key': 'True'})
        },
        'plumbing.collecttimed': {
            'Meta': {'object_name': 'CollectTimed', '_ormbases': ['factopy.Collect']},
            u'collect_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['factopy.Collect']", 'unique': 'True', 'primary_key': 'True'}),
            'daily': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'hourly': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'monthly': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'slotly': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'slots_by_day': ('django.db.models.fields.IntegerField', [], {'default': '1'}),
            'week_day': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'weekly': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'yearly': ('django.db.models.fields.BooleanField', [], {'default': 'False'})
        },
        'plumbing.compact': {
            'Meta': {'object_name': 'Compact', '_ormbases': ['factopy.Adapt']},
            u'adapt_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['factopy.Adapt']", 'unique': 'True', 'primary_key': 'True'}),
            'extension': ('django.db.models.fields.TextField', [], {})
        },
        'plumbing.filterchannel': {
            'Meta': {'object_name': 'FilterChannel', '_ormbases': ['factopy.Filter']},
            'channels': ('django.db.models.fields.related.ManyToManyField', [], {'to': "orm['requester.Channel']", 'db_index': 'True', 'symmetrical': 'False'}),
            u'filter_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['factopy.Filter']", 'unique': 'True', 'primary_key': 'True'})
        },
        'plumbing.filtersolarelevation': {
            'Meta': {'object_name': 'FilterSolarElevation', '_ormbases': ['factopy.Filter']},
            u'filter_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['factopy.Filter']", 'unique': 'True', 'primary_key': 'True'}),
            'hourly_longitude': ('django.db.models.fields.DecimalField', [], {'default': "'0.00'", 'max_digits': '5', 'decimal_places': '2'}),
            'minimum': ('django.db.models.fields.DecimalField', [], {'max_digits': '4', 'decimal_places': '2'})
        },
        'plumbing.filtertimed': {
            'Meta': {'object_name': 'FilterTimed', '_ormbases': ['factopy.Filter']},
            'daily': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'error': ('django.db.models.fields.IntegerField', [], {'default': '0'}),
            u'filter_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['factopy.Filter']", 'unique': 'True', 'primary_key': 'True'}),
            'hourly': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'monthly': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'number': ('django.db.models.fields.IntegerField', [], {'default': '0'}),
            'slotly': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'slots_by_day': ('django.db.models.fields.IntegerField', [], {'default': '1'}),
            'week_day': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'weekly': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'yearly': ('django.db.models.fields.BooleanField', [], {'default': 'False'})
        },
        'plumbing.image': {
            'Meta': {'object_name': 'Image', '_ormbases': ['requester.File']},
            u'file_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['requester.File']", 'unique': 'True', 'primary_key': 'True'})
        },
        'plumbing.program': {
            'Meta': {'object_name': 'Program', '_ormbases': ['factopy.ComplexProcess']},
            u'complexprocess_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['factopy.ComplexProcess']", 'unique': 'True', 'primary_key': 'True'}),
            'stream': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['factopy.Stream']"})
        },
        'plumbing.syncimporter': {
            'Meta': {'object_name': 'SyncImporter', '_ormbases': ['factopy.Importer']},
            u'importer_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['factopy.Importer']", 'unique': 'True', 'primary_key': 'True'})
        },
        'requester.account': {
            'Meta': {'object_name': 'Account'},
            'created': ('django.db.models.fields.DateTimeField', [], {'auto_now_add': 'True', 'blank': 'True'}),
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'modified': ('django.db.models.fields.DateTimeField', [], {'auto_now': 'True', 'blank': 'True'}),
            'password': ('django.db.models.fields.TextField', [], {}),
            'polymorphic_ctype': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'polymorphic_requester.account_set'", 'null': 'True', 'to': u"orm['contenttypes.ContentType']"})
        },
        'requester.area': {
            'Meta': {'object_name': 'Area'},
            'created': ('django.db.models.fields.DateTimeField', [], {'auto_now_add': 'True', 'blank': 'True'}),
            'east_longitude': ('django.db.models.fields.DecimalField', [], {'max_digits': '5', 'decimal_places': '2'}),
            'hourly_longitude': ('django.db.models.fields.DecimalField', [], {'default': "'0.00'", 'max_digits': '5', 'decimal_places': '2'}),
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'modified': ('django.db.models.fields.DateTimeField', [], {'auto_now': 'True', 'blank': 'True'}),
            'name': ('django.db.models.fields.TextField', [], {}),
            'north_latitude': ('django.db.models.fields.DecimalField', [], {'max_digits': '4', 'decimal_places': '2'}),
            'south_latitude': ('django.db.models.fields.DecimalField', [], {'max_digits': '4', 'decimal_places': '2'}),
            'west_longitude': ('django.db.models.fields.DecimalField', [], {'max_digits': '5', 'decimal_places': '2'})
        },
        'requester.automaticdownload': {
            'Meta': {'object_name': 'AutomaticDownload'},
            'area': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['requester.Area']"}),
            'channels': ('django.db.models.fields.related.ManyToManyField', [], {'to': "orm['requester.Channel']", 'symmetrical': 'False'}),
            'created': ('django.db.models.fields.DateTimeField', [], {'auto_now_add': 'True', 'blank': 'True'}),
            'email_server': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['requester.EmailAccount']"}),
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'max_simultaneous_request': ('django.db.models.fields.IntegerField', [], {}),
            'modified': ('django.db.models.fields.DateTimeField', [], {'auto_now': 'True', 'blank': 'True'}),
            'paused': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'root_path': ('django.db.models.fields.TextField', [], {}),
            'time_range': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['requester.UTCTimeRange']"}),
            'title': ('django.db.models.fields.TextField', [], {'db_index': 'True'})
        },
        'requester.channel': {
            'Meta': {'object_name': 'Channel'},
            'created': ('django.db.models.fields.DateTimeField', [], {'auto_now_add': 'True', 'blank': 'True'}),
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'in_file': ('django.db.models.fields.TextField', [], {'null': 'True', 'db_index': 'True'}),
            'modified': ('django.db.models.fields.DateTimeField', [], {'auto_now': 'True', 'blank': 'True'}),
            'name': ('django.db.models.fields.TextField', [], {'db_index': 'True'}),
            'satellite': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['requester.Satellite']"})
        },
        'requester.emailaccount': {
            'Meta': {'object_name': 'EmailAccount', '_ormbases': ['requester.Account']},
            u'account_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['requester.Account']", 'unique': 'True', 'primary_key': 'True'}),
            'hostname': ('django.db.models.fields.TextField', [], {}),
            'port': ('django.db.models.fields.IntegerField', [], {}),
            'username': ('django.db.models.fields.EmailField', [], {'max_length': '75'})
        },
        'requester.file': {
            'Meta': {'object_name': 'File', '_ormbases': ['factopy.Material']},
            'begin_download': ('django.db.models.fields.DateTimeField', [], {'null': 'True', 'db_index': 'True'}),
            'downloaded': ('django.db.models.fields.BooleanField', [], {'default': 'False', 'db_index': 'True'}),
            'end_download': ('django.db.models.fields.DateTimeField', [], {'null': 'True', 'db_index': 'True'}),
            'failures': ('django.db.models.fields.IntegerField', [], {'default': '0'}),
            'localname': ('django.db.models.fields.TextField', [], {'default': "''", 'unique': 'True', 'db_index': 'True'}),
            u'material_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['factopy.Material']", 'unique': 'True', 'primary_key': 'True'}),
            'order': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['requester.Order']", 'null': 'True'}),
            'remotename': ('django.db.models.fields.TextField', [], {'null': 'True'}),
            'size': ('django.db.models.fields.IntegerField', [], {'null': 'True'})
        },
        'requester.ftpserveraccount': {
            'Meta': {'object_name': 'FTPServerAccount', '_ormbases': ['requester.ServerAccount']},
            'hostname': ('django.db.models.fields.TextField', [], {}),
            u'serveraccount_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['requester.ServerAccount']", 'unique': 'True', 'primary_key': 'True'})
        },
        'requester.order': {
            'Meta': {'object_name': 'Order', '_ormbases': ['factopy.Material']},
            'downloaded': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'empty_flag': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'identification': ('django.db.models.fields.TextField', [], {'db_index': 'True'}),
            u'material_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['factopy.Material']", 'unique': 'True', 'primary_key': 'True'}),
            'request': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['requester.Request']", 'unique': 'True'}),
            'server': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['requester.FTPServerAccount']", 'null': 'True'})
        },
        'requester.request': {
            'Meta': {'object_name': 'Request', '_ormbases': ['factopy.Material']},
            'aged': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'automatic_download': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['requester.AutomaticDownload']"}),
            'begin': ('django.db.models.fields.DateTimeField', [], {'db_index': 'True'}),
            'end': ('django.db.models.fields.DateTimeField', [], {'db_index': 'True'}),
            u'material_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['factopy.Material']", 'unique': 'True', 'primary_key': 'True'})
        },
        'requester.satellite': {
            'Meta': {'object_name': 'Satellite'},
            'created': ('django.db.models.fields.DateTimeField', [], {'auto_now_add': 'True', 'blank': 'True'}),
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'identification': ('django.db.models.fields.TextField', [], {}),
            'in_file': ('django.db.models.fields.TextField', [], {}),
            'modified': ('django.db.models.fields.DateTimeField', [], {'auto_now': 'True', 'blank': 'True'}),
            'name': ('django.db.models.fields.TextField', [], {}),
            'request_server': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['requester.WebServerAccount']"})
        },
        'requester.serveraccount': {
            'Meta': {'object_name': 'ServerAccount', '_ormbases': ['requester.Account']},
            u'account_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['requester.Account']", 'unique': 'True', 'primary_key': 'True'}),
            'username': ('django.db.models.fields.TextField', [], {})
        },
        'requester.utctimerange': {
            'Meta': {'object_name': 'UTCTimeRange'},
            'begin': ('django.db.models.fields.DateTimeField', [], {}),
            'created': ('django.db.models.fields.DateTimeField', [], {'auto_now_add': 'True', 'blank': 'True'}),
            'end': ('django.db.models.fields.DateTimeField', [], {}),
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'modified': ('django.db.models.fields.DateTimeField', [], {'auto_now': 'True', 'blank': 'True'})
        },
        'requester.webserveraccount': {
            'Meta': {'object_name': 'WebServerAccount', '_ormbases': ['requester.ServerAccount']},
            u'serveraccount_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['requester.ServerAccount']", 'unique': 'True', 'primary_key': 'True'}),
            'url': ('django.db.models.fields.TextField', [], {})
        }
    }

    complete_apps = ['plumbing']