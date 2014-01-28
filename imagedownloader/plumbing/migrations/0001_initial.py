# -*- coding: utf-8 -*-
import datetime
from south.db import db
from south.v2 import SchemaMigration
from django.db import models


class Migration(SchemaMigration):

    def forwards(self, orm):
        # Adding model 'TagManager'
        db.create_table(u'plumbing_tagmanager', (
            (u'id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('tag_string', self.gf('django.db.models.fields.TextField')(default='', db_index=True)),
        ))
        db.send_create_signal('plumbing', ['TagManager'])

        # Adding model 'Stream'
        db.create_table(u'plumbing_stream', (
            (u'id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('root_path', self.gf('django.db.models.fields.TextField')(db_index=True)),
            ('tags', self.gf('django.db.models.fields.related.ForeignKey')(related_name='stream', to=orm['plumbing.TagManager'])),
            ('created', self.gf('django.db.models.fields.DateTimeField')(default=datetime.datetime(2014, 1, 28, 0, 0))),
            ('modified', self.gf('django.db.models.fields.DateTimeField')(default=datetime.datetime(2014, 1, 28, 0, 0))),
        ))
        db.send_create_signal('plumbing', ['Stream'])

        # Adding model 'Material'
        db.create_table(u'plumbing_material', (
            (u'id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('polymorphic_ctype', self.gf('django.db.models.fields.related.ForeignKey')(related_name='polymorphic_plumbing.material_set', null=True, to=orm['contenttypes.ContentType'])),
            ('created', self.gf('django.db.models.fields.DateTimeField')(auto_now_add=True, blank=True)),
            ('modified', self.gf('django.db.models.fields.DateTimeField')(auto_now=True, blank=True)),
        ))
        db.send_create_signal('plumbing', ['Material'])

        # Adding model 'MaterialStatus'
        db.create_table(u'plumbing_materialstatus', (
            (u'id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('material', self.gf('django.db.models.fields.related.ForeignKey')(related_name='stream', to=orm['plumbing.Material'])),
            ('stream', self.gf('django.db.models.fields.related.ForeignKey')(related_name='materials', to=orm['plumbing.Stream'])),
            ('processed', self.gf('django.db.models.fields.BooleanField')(default=False)),
        ))
        db.send_create_signal('plumbing', ['MaterialStatus'])

        # Adding unique constraint on 'MaterialStatus', fields ['material', 'stream']
        db.create_unique(u'plumbing_materialstatus', ['material_id', 'stream_id'])

        # Adding model 'Process'
        db.create_table(u'plumbing_process', (
            (u'id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('polymorphic_ctype', self.gf('django.db.models.fields.related.ForeignKey')(related_name='polymorphic_plumbing.process_set', null=True, to=orm['contenttypes.ContentType'])),
            ('name', self.gf('django.db.models.fields.TextField')(db_index=True)),
            ('description', self.gf('django.db.models.fields.TextField')(db_index=True)),
        ))
        db.send_create_signal('plumbing', ['Process'])

        # Adding model 'ComplexProcess'
        db.create_table(u'plumbing_complexprocess', (
            (u'process_ptr', self.gf('django.db.models.fields.related.OneToOneField')(to=orm['plumbing.Process'], unique=True, primary_key=True)),
        ))
        db.send_create_signal('plumbing', ['ComplexProcess'])

        # Adding model 'ProcessOrder'
        db.create_table(u'plumbing_processorder', (
            (u'id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('position', self.gf('django.db.models.fields.IntegerField')()),
            ('process', self.gf('django.db.models.fields.related.ForeignKey')(related_name='used_by', to=orm['plumbing.Process'])),
            ('complex_process', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['plumbing.ComplexProcess'])),
        ))
        db.send_create_signal('plumbing', ['ProcessOrder'])

        # Adding model 'Collect'
        db.create_table(u'plumbing_collect', (
            (u'process_ptr', self.gf('django.db.models.fields.related.OneToOneField')(to=orm['plumbing.Process'], unique=True, primary_key=True)),
        ))
        db.send_create_signal('plumbing', ['Collect'])

        # Adding model 'Filter'
        db.create_table(u'plumbing_filter', (
            (u'process_ptr', self.gf('django.db.models.fields.related.OneToOneField')(to=orm['plumbing.Process'], unique=True, primary_key=True)),
        ))
        db.send_create_signal('plumbing', ['Filter'])

        # Adding model 'Adapter'
        db.create_table(u'plumbing_adapter', (
            (u'process_ptr', self.gf('django.db.models.fields.related.OneToOneField')(to=orm['plumbing.Process'], unique=True, primary_key=True)),
            ('stream', self.gf('django.db.models.fields.related.ForeignKey')(default=None, to=orm['plumbing.Stream'], null=True)),
        ))
        db.send_create_signal('plumbing', ['Adapter'])

        # Adding model 'Importer'
        db.create_table(u'plumbing_importer', (
            (u'adapter_ptr', self.gf('django.db.models.fields.related.OneToOneField')(to=orm['plumbing.Adapter'], unique=True, primary_key=True)),
            ('frequency', self.gf('django.db.models.fields.IntegerField')(default=900)),
        ))
        db.send_create_signal('plumbing', ['Importer'])

        # Adding model 'File'
        db.create_table(u'plumbing_file', (
            (u'material_ptr', self.gf('django.db.models.fields.related.OneToOneField')(to=orm['plumbing.Material'], unique=True, primary_key=True)),
            ('localname', self.gf('django.db.models.fields.TextField')(default='', unique=True, db_index=True)),
        ))
        db.send_create_signal('plumbing', ['File'])

        # Adding model 'Image'
        db.create_table(u'plumbing_image', (
            (u'file_ptr', self.gf('django.db.models.fields.related.OneToOneField')(to=orm['plumbing.File'], unique=True, primary_key=True)),
        ))
        db.send_create_signal('plumbing', ['Image'])

        # Adding model 'CollectTimed'
        db.create_table(u'plumbing_collecttimed', (
            (u'collect_ptr', self.gf('django.db.models.fields.related.OneToOneField')(to=orm['plumbing.Collect'], unique=True, primary_key=True)),
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
            (u'collect_ptr', self.gf('django.db.models.fields.related.OneToOneField')(to=orm['plumbing.Collect'], unique=True, primary_key=True)),
        ))
        db.send_create_signal('plumbing', ['CollectChannel'])

        # Adding model 'FilterChannel'
        db.create_table(u'plumbing_filterchannel', (
            (u'filter_ptr', self.gf('django.db.models.fields.related.OneToOneField')(to=orm['plumbing.Filter'], unique=True, primary_key=True)),
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
            (u'filter_ptr', self.gf('django.db.models.fields.related.OneToOneField')(to=orm['plumbing.Filter'], unique=True, primary_key=True)),
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
            (u'filter_ptr', self.gf('django.db.models.fields.related.OneToOneField')(to=orm['plumbing.Filter'], unique=True, primary_key=True)),
            ('minimum', self.gf('django.db.models.fields.DecimalField')(max_digits=4, decimal_places=2)),
            ('hourly_longitude', self.gf('django.db.models.fields.DecimalField')(default='0.00', max_digits=5, decimal_places=2)),
        ))
        db.send_create_signal('plumbing', ['FilterSolarElevation'])

        # Adding model 'AppendCountToRadiationCoefficient'
        db.create_table(u'plumbing_appendcounttoradiationcoefficient', (
            (u'process_ptr', self.gf('django.db.models.fields.related.OneToOneField')(to=orm['plumbing.Process'], unique=True, primary_key=True)),
        ))
        db.send_create_signal('plumbing', ['AppendCountToRadiationCoefficient'])

        # Adding model 'SyncImporter'
        db.create_table(u'plumbing_syncimporter', (
            (u'importer_ptr', self.gf('django.db.models.fields.related.OneToOneField')(to=orm['plumbing.Importer'], unique=True, primary_key=True)),
        ))
        db.send_create_signal('plumbing', ['SyncImporter'])

        # Adding model 'Program'
        db.create_table(u'plumbing_program', (
            (u'complexprocess_ptr', self.gf('django.db.models.fields.related.OneToOneField')(to=orm['plumbing.ComplexProcess'], unique=True, primary_key=True)),
            ('stream', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['plumbing.Stream'])),
        ))
        db.send_create_signal('plumbing', ['Program'])

        # Adding model 'Compact'
        db.create_table(u'plumbing_compact', (
            (u'adapter_ptr', self.gf('django.db.models.fields.related.OneToOneField')(to=orm['plumbing.Adapter'], unique=True, primary_key=True)),
            ('extension', self.gf('django.db.models.fields.TextField')()),
        ))
        db.send_create_signal('plumbing', ['Compact'])


    def backwards(self, orm):
        # Removing unique constraint on 'MaterialStatus', fields ['material', 'stream']
        db.delete_unique(u'plumbing_materialstatus', ['material_id', 'stream_id'])

        # Deleting model 'TagManager'
        db.delete_table(u'plumbing_tagmanager')

        # Deleting model 'Stream'
        db.delete_table(u'plumbing_stream')

        # Deleting model 'Material'
        db.delete_table(u'plumbing_material')

        # Deleting model 'MaterialStatus'
        db.delete_table(u'plumbing_materialstatus')

        # Deleting model 'Process'
        db.delete_table(u'plumbing_process')

        # Deleting model 'ComplexProcess'
        db.delete_table(u'plumbing_complexprocess')

        # Deleting model 'ProcessOrder'
        db.delete_table(u'plumbing_processorder')

        # Deleting model 'Collect'
        db.delete_table(u'plumbing_collect')

        # Deleting model 'Filter'
        db.delete_table(u'plumbing_filter')

        # Deleting model 'Adapter'
        db.delete_table(u'plumbing_adapter')

        # Deleting model 'Importer'
        db.delete_table(u'plumbing_importer')

        # Deleting model 'File'
        db.delete_table(u'plumbing_file')

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
        'plumbing.adapter': {
            'Meta': {'object_name': 'Adapter', '_ormbases': ['plumbing.Process']},
            u'process_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['plumbing.Process']", 'unique': 'True', 'primary_key': 'True'}),
            'stream': ('django.db.models.fields.related.ForeignKey', [], {'default': 'None', 'to': "orm['plumbing.Stream']", 'null': 'True'})
        },
        'plumbing.appendcounttoradiationcoefficient': {
            'Meta': {'object_name': 'AppendCountToRadiationCoefficient', '_ormbases': ['plumbing.Process']},
            u'process_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['plumbing.Process']", 'unique': 'True', 'primary_key': 'True'})
        },
        'plumbing.collect': {
            'Meta': {'object_name': 'Collect', '_ormbases': ['plumbing.Process']},
            u'process_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['plumbing.Process']", 'unique': 'True', 'primary_key': 'True'})
        },
        'plumbing.collectchannel': {
            'Meta': {'object_name': 'CollectChannel', '_ormbases': ['plumbing.Collect']},
            u'collect_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['plumbing.Collect']", 'unique': 'True', 'primary_key': 'True'})
        },
        'plumbing.collecttimed': {
            'Meta': {'object_name': 'CollectTimed', '_ormbases': ['plumbing.Collect']},
            u'collect_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['plumbing.Collect']", 'unique': 'True', 'primary_key': 'True'}),
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
            'Meta': {'object_name': 'Compact', '_ormbases': ['plumbing.Adapter']},
            u'adapter_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['plumbing.Adapter']", 'unique': 'True', 'primary_key': 'True'}),
            'extension': ('django.db.models.fields.TextField', [], {})
        },
        'plumbing.complexprocess': {
            'Meta': {'object_name': 'ComplexProcess', '_ormbases': ['plumbing.Process']},
            u'process_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['plumbing.Process']", 'unique': 'True', 'primary_key': 'True'}),
            'processes': ('django.db.models.fields.related.ManyToManyField', [], {'related_name': "'complex_process'", 'symmetrical': 'False', 'through': "orm['plumbing.ProcessOrder']", 'to': "orm['plumbing.Process']"})
        },
        'plumbing.file': {
            'Meta': {'object_name': 'File', '_ormbases': ['plumbing.Material']},
            'localname': ('django.db.models.fields.TextField', [], {'default': "''", 'unique': 'True', 'db_index': 'True'}),
            u'material_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['plumbing.Material']", 'unique': 'True', 'primary_key': 'True'})
        },
        'plumbing.filter': {
            'Meta': {'object_name': 'Filter', '_ormbases': ['plumbing.Process']},
            u'process_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['plumbing.Process']", 'unique': 'True', 'primary_key': 'True'})
        },
        'plumbing.filterchannel': {
            'Meta': {'object_name': 'FilterChannel', '_ormbases': ['plumbing.Filter']},
            'channels': ('django.db.models.fields.related.ManyToManyField', [], {'to': "orm['requester.Channel']", 'db_index': 'True', 'symmetrical': 'False'}),
            u'filter_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['plumbing.Filter']", 'unique': 'True', 'primary_key': 'True'})
        },
        'plumbing.filtersolarelevation': {
            'Meta': {'object_name': 'FilterSolarElevation', '_ormbases': ['plumbing.Filter']},
            u'filter_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['plumbing.Filter']", 'unique': 'True', 'primary_key': 'True'}),
            'hourly_longitude': ('django.db.models.fields.DecimalField', [], {'default': "'0.00'", 'max_digits': '5', 'decimal_places': '2'}),
            'minimum': ('django.db.models.fields.DecimalField', [], {'max_digits': '4', 'decimal_places': '2'})
        },
        'plumbing.filtertimed': {
            'Meta': {'object_name': 'FilterTimed', '_ormbases': ['plumbing.Filter']},
            'daily': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'error': ('django.db.models.fields.IntegerField', [], {'default': '0'}),
            u'filter_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['plumbing.Filter']", 'unique': 'True', 'primary_key': 'True'}),
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
            'Meta': {'object_name': 'Image', '_ormbases': ['plumbing.File']},
            u'file_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['plumbing.File']", 'unique': 'True', 'primary_key': 'True'})
        },
        'plumbing.importer': {
            'Meta': {'object_name': 'Importer', '_ormbases': ['plumbing.Adapter']},
            u'adapter_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['plumbing.Adapter']", 'unique': 'True', 'primary_key': 'True'}),
            'frequency': ('django.db.models.fields.IntegerField', [], {'default': '900'})
        },
        'plumbing.material': {
            'Meta': {'object_name': 'Material'},
            'created': ('django.db.models.fields.DateTimeField', [], {'auto_now_add': 'True', 'blank': 'True'}),
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'modified': ('django.db.models.fields.DateTimeField', [], {'auto_now': 'True', 'blank': 'True'}),
            'polymorphic_ctype': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'polymorphic_plumbing.material_set'", 'null': 'True', 'to': u"orm['contenttypes.ContentType']"})
        },
        'plumbing.materialstatus': {
            'Meta': {'unique_together': "(('material', 'stream'),)", 'object_name': 'MaterialStatus'},
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'material': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'stream'", 'to': "orm['plumbing.Material']"}),
            'processed': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'stream': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'materials'", 'to': "orm['plumbing.Stream']"})
        },
        'plumbing.process': {
            'Meta': {'object_name': 'Process'},
            'description': ('django.db.models.fields.TextField', [], {'db_index': 'True'}),
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'name': ('django.db.models.fields.TextField', [], {'db_index': 'True'}),
            'polymorphic_ctype': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'polymorphic_plumbing.process_set'", 'null': 'True', 'to': u"orm['contenttypes.ContentType']"})
        },
        'plumbing.processorder': {
            'Meta': {'object_name': 'ProcessOrder'},
            'complex_process': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['plumbing.ComplexProcess']"}),
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'position': ('django.db.models.fields.IntegerField', [], {}),
            'process': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'used_by'", 'to': "orm['plumbing.Process']"})
        },
        'plumbing.program': {
            'Meta': {'object_name': 'Program', '_ormbases': ['plumbing.ComplexProcess']},
            u'complexprocess_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['plumbing.ComplexProcess']", 'unique': 'True', 'primary_key': 'True'}),
            'stream': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['plumbing.Stream']"})
        },
        'plumbing.stream': {
            'Meta': {'object_name': 'Stream'},
            'created': ('django.db.models.fields.DateTimeField', [], {'default': 'datetime.datetime(2014, 1, 28, 0, 0)'}),
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'modified': ('django.db.models.fields.DateTimeField', [], {'default': 'datetime.datetime(2014, 1, 28, 0, 0)'}),
            'root_path': ('django.db.models.fields.TextField', [], {'db_index': 'True'}),
            'tags': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'stream'", 'to': "orm['plumbing.TagManager']"})
        },
        'plumbing.syncimporter': {
            'Meta': {'object_name': 'SyncImporter', '_ormbases': ['plumbing.Importer']},
            u'importer_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['plumbing.Importer']", 'unique': 'True', 'primary_key': 'True'})
        },
        'plumbing.tagmanager': {
            'Meta': {'object_name': 'TagManager'},
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'tag_string': ('django.db.models.fields.TextField', [], {'default': "''", 'db_index': 'True'})
        },
        'requester.account': {
            'Meta': {'object_name': 'Account'},
            'created': ('django.db.models.fields.DateTimeField', [], {'auto_now_add': 'True', 'blank': 'True'}),
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'modified': ('django.db.models.fields.DateTimeField', [], {'auto_now': 'True', 'blank': 'True'}),
            'password': ('django.db.models.fields.TextField', [], {}),
            'polymorphic_ctype': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'polymorphic_requester.account_set'", 'null': 'True', 'to': u"orm['contenttypes.ContentType']"})
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
        'requester.webserveraccount': {
            'Meta': {'object_name': 'WebServerAccount', '_ormbases': ['requester.ServerAccount']},
            u'serveraccount_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['requester.ServerAccount']", 'unique': 'True', 'primary_key': 'True'}),
            'url': ('django.db.models.fields.TextField', [], {})
        }
    }

    complete_apps = ['plumbing']