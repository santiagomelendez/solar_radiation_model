# -*- coding: utf-8 -*-
import datetime
from south.db import db
from south.v2 import SchemaMigration
from django.db import models


class Migration(SchemaMigration):

    def forwards(self, orm):
        # Adding model 'Area'
        db.create_table(u'requester_area', (
            (u'id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('name', self.gf('django.db.models.fields.TextField')()),
            ('north_latitude', self.gf('django.db.models.fields.DecimalField')(max_digits=4, decimal_places=2)),
            ('south_latitude', self.gf('django.db.models.fields.DecimalField')(max_digits=4, decimal_places=2)),
            ('east_longitude', self.gf('django.db.models.fields.DecimalField')(max_digits=5, decimal_places=2)),
            ('west_longitude', self.gf('django.db.models.fields.DecimalField')(max_digits=5, decimal_places=2)),
            ('hourly_longitude', self.gf('django.db.models.fields.DecimalField')(default='0.00', max_digits=5, decimal_places=2)),
            ('created', self.gf('django.db.models.fields.DateTimeField')(auto_now_add=True, blank=True)),
            ('modified', self.gf('django.db.models.fields.DateTimeField')(auto_now=True, blank=True)),
        ))
        db.send_create_signal('requester', ['Area'])

        # Adding model 'Satellite'
        db.create_table(u'requester_satellite', (
            (u'id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('name', self.gf('django.db.models.fields.TextField')()),
            ('identification', self.gf('django.db.models.fields.TextField')()),
            ('in_file', self.gf('django.db.models.fields.TextField')()),
            ('created', self.gf('django.db.models.fields.DateTimeField')(auto_now_add=True, blank=True)),
            ('modified', self.gf('django.db.models.fields.DateTimeField')(auto_now=True, blank=True)),
        ))
        db.send_create_signal('requester', ['Satellite'])

        # Adding model 'Channel'
        db.create_table(u'requester_channel', (
            (u'id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('name', self.gf('django.db.models.fields.TextField')(db_index=True)),
            ('in_file', self.gf('django.db.models.fields.TextField')(null=True, db_index=True)),
            ('satellite', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['requester.Satellite'])),
            ('created', self.gf('django.db.models.fields.DateTimeField')(auto_now_add=True, blank=True)),
            ('modified', self.gf('django.db.models.fields.DateTimeField')(auto_now=True, blank=True)),
        ))
        db.send_create_signal('requester', ['Channel'])

        # Adding model 'Account'
        db.create_table(u'requester_account', (
            (u'id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('polymorphic_ctype', self.gf('django.db.models.fields.related.ForeignKey')(related_name='polymorphic_requester.account_set', null=True, to=orm['contenttypes.ContentType'])),
            ('password', self.gf('django.db.models.fields.TextField')()),
            ('created', self.gf('django.db.models.fields.DateTimeField')(auto_now_add=True, blank=True)),
            ('modified', self.gf('django.db.models.fields.DateTimeField')(auto_now=True, blank=True)),
        ))
        db.send_create_signal('requester', ['Account'])

        # Adding model 'EmailAccount'
        db.create_table(u'requester_emailaccount', (
            (u'account_ptr', self.gf('django.db.models.fields.related.OneToOneField')(to=orm['requester.Account'], unique=True, primary_key=True)),
            ('hostname', self.gf('django.db.models.fields.TextField')()),
            ('port', self.gf('django.db.models.fields.IntegerField')()),
            ('username', self.gf('django.db.models.fields.EmailField')(max_length=75)),
        ))
        db.send_create_signal('requester', ['EmailAccount'])

        # Adding model 'ServerAccount'
        db.create_table(u'requester_serveraccount', (
            (u'account_ptr', self.gf('django.db.models.fields.related.OneToOneField')(to=orm['requester.Account'], unique=True, primary_key=True)),
            ('username', self.gf('django.db.models.fields.TextField')()),
        ))
        db.send_create_signal('requester', ['ServerAccount'])

        # Adding model 'WebServerAccount'
        db.create_table(u'requester_webserveraccount', (
            (u'serveraccount_ptr', self.gf('django.db.models.fields.related.OneToOneField')(to=orm['requester.ServerAccount'], unique=True, primary_key=True)),
            ('url', self.gf('django.db.models.fields.TextField')()),
        ))
        db.send_create_signal('requester', ['WebServerAccount'])

        # Adding model 'FTPServerAccount'
        db.create_table(u'requester_ftpserveraccount', (
            (u'serveraccount_ptr', self.gf('django.db.models.fields.related.OneToOneField')(to=orm['requester.ServerAccount'], unique=True, primary_key=True)),
            ('hostname', self.gf('django.db.models.fields.TextField')()),
        ))
        db.send_create_signal('requester', ['FTPServerAccount'])

        # Adding model 'NOAAAdapt'
        db.create_table(u'requester_noaaadapt', (
            (u'adapt_ptr', self.gf('django.db.models.fields.related.OneToOneField')(to=orm['factopy.Adapt'], unique=True, primary_key=True)),
            ('title', self.gf('django.db.models.fields.TextField')(db_index=True)),
            ('area', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['requester.Area'])),
            ('paused', self.gf('django.db.models.fields.BooleanField')(default=False)),
            ('max_simultaneous_request', self.gf('django.db.models.fields.IntegerField')()),
            ('email_server', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['requester.EmailAccount'])),
            ('request_server', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['requester.WebServerAccount'])),
            ('root_path', self.gf('django.db.models.fields.TextField')()),
            ('created', self.gf('django.db.models.fields.DateTimeField')(auto_now_add=True, blank=True)),
            ('modified', self.gf('django.db.models.fields.DateTimeField')(auto_now=True, blank=True)),
            ('begin', self.gf('django.db.models.fields.DateTimeField')(null=True)),
            ('end', self.gf('django.db.models.fields.DateTimeField')(null=True)),
        ))
        db.send_create_signal('requester', ['NOAAAdapt'])

        # Adding M2M table for field channels on 'NOAAAdapt'
        db.create_table(u'requester_noaaadapt_channels', (
            ('id', models.AutoField(verbose_name='ID', primary_key=True, auto_created=True)),
            ('noaaadapt', models.ForeignKey(orm['requester.noaaadapt'], null=False)),
            ('channel', models.ForeignKey(orm['requester.channel'], null=False))
        ))
        db.create_unique(u'requester_noaaadapt_channels', ['noaaadapt_id', 'channel_id'])

        # Adding model 'Job'
        db.create_table(u'requester_job', (
            (u'process_ptr', self.gf('django.db.models.fields.related.OneToOneField')(to=orm['factopy.Process'], unique=True, primary_key=True)),
        ))
        db.send_create_signal('requester', ['Job'])

        # Adding model 'NOAAEMailChecker'
        db.create_table(u'requester_noaaemailchecker', (
            (u'job_ptr', self.gf('django.db.models.fields.related.OneToOneField')(to=orm['requester.Job'], unique=True, primary_key=True)),
            ('server', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['requester.EmailAccount'])),
        ))
        db.send_create_signal('requester', ['NOAAEMailChecker'])

        # Adding model 'NOAAFTPDownloader'
        db.create_table(u'requester_noaaftpdownloader', (
            (u'job_ptr', self.gf('django.db.models.fields.related.OneToOneField')(to=orm['requester.Job'], unique=True, primary_key=True)),
            ('service', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['requester.FTPServerAccount'], null=True)),
            ('file', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['requester.File'], null=True)),
        ))
        db.send_create_signal('requester', ['NOAAFTPDownloader'])

        # Adding model 'GOESRequest'
        db.create_table(u'requester_goesrequest', (
            (u'job_ptr', self.gf('django.db.models.fields.related.OneToOneField')(to=orm['requester.Job'], unique=True, primary_key=True)),
        ))
        db.send_create_signal('requester', ['GOESRequest'])

        # Adding model 'QOSRequester'
        db.create_table(u'requester_qosrequester', (
            (u'job_ptr', self.gf('django.db.models.fields.related.OneToOneField')(to=orm['requester.Job'], unique=True, primary_key=True)),
        ))
        db.send_create_signal('requester', ['QOSRequester'])

        # Adding model 'QOSManager__impl'
        db.create_table(u'requester_qosmanager__impl', (
            (u'job_ptr', self.gf('django.db.models.fields.related.OneToOneField')(to=orm['requester.Job'], unique=True, primary_key=True)),
        ))
        db.send_create_signal('requester', ['QOSManager__impl'])

        # Adding model 'Request'
        db.create_table(u'requester_request', (
            (u'material_ptr', self.gf('django.db.models.fields.related.OneToOneField')(to=orm['factopy.Material'], unique=True, primary_key=True)),
            ('adapt', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['requester.NOAAAdapt'])),
            ('begin', self.gf('django.db.models.fields.DateTimeField')(db_index=True)),
            ('end', self.gf('django.db.models.fields.DateTimeField')(db_index=True)),
            ('aged', self.gf('django.db.models.fields.BooleanField')(default=False)),
        ))
        db.send_create_signal('requester', ['Request'])

        # Adding model 'File'
        db.create_table(u'requester_file', (
            (u'material_ptr', self.gf('django.db.models.fields.related.OneToOneField')(to=orm['factopy.Material'], unique=True, primary_key=True)),
            ('localname', self.gf('django.db.models.fields.TextField')(default='', unique=True, db_index=True)),
            ('remotename', self.gf('django.db.models.fields.TextField')(null=True)),
            ('size', self.gf('django.db.models.fields.IntegerField')(null=True)),
            ('downloaded', self.gf('django.db.models.fields.BooleanField')(default=False, db_index=True)),
            ('begin_download', self.gf('django.db.models.fields.DateTimeField')(null=True, db_index=True)),
            ('end_download', self.gf('django.db.models.fields.DateTimeField')(null=True, db_index=True)),
            ('failures', self.gf('django.db.models.fields.IntegerField')(default=0)),
        ))
        db.send_create_signal('requester', ['File'])

        # Adding model 'Image'
        db.create_table(u'requester_image', (
            (u'file_ptr', self.gf('django.db.models.fields.related.OneToOneField')(to=orm['requester.File'], unique=True, primary_key=True)),
        ))
        db.send_create_signal('requester', ['Image'])


    def backwards(self, orm):
        # Deleting model 'Area'
        db.delete_table(u'requester_area')

        # Deleting model 'Satellite'
        db.delete_table(u'requester_satellite')

        # Deleting model 'Channel'
        db.delete_table(u'requester_channel')

        # Deleting model 'Account'
        db.delete_table(u'requester_account')

        # Deleting model 'EmailAccount'
        db.delete_table(u'requester_emailaccount')

        # Deleting model 'ServerAccount'
        db.delete_table(u'requester_serveraccount')

        # Deleting model 'WebServerAccount'
        db.delete_table(u'requester_webserveraccount')

        # Deleting model 'FTPServerAccount'
        db.delete_table(u'requester_ftpserveraccount')

        # Deleting model 'NOAAAdapt'
        db.delete_table(u'requester_noaaadapt')

        # Removing M2M table for field channels on 'NOAAAdapt'
        db.delete_table('requester_noaaadapt_channels')

        # Deleting model 'Job'
        db.delete_table(u'requester_job')

        # Deleting model 'NOAAEMailChecker'
        db.delete_table(u'requester_noaaemailchecker')

        # Deleting model 'NOAAFTPDownloader'
        db.delete_table(u'requester_noaaftpdownloader')

        # Deleting model 'GOESRequest'
        db.delete_table(u'requester_goesrequest')

        # Deleting model 'QOSRequester'
        db.delete_table(u'requester_qosrequester')

        # Deleting model 'QOSManager__impl'
        db.delete_table(u'requester_qosmanager__impl')

        # Deleting model 'Request'
        db.delete_table(u'requester_request')

        # Deleting model 'File'
        db.delete_table(u'requester_file')

        # Deleting model 'Image'
        db.delete_table(u'requester_image')


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
        'factopy.stream': {
            'Meta': {'object_name': 'Stream'},
            'created': ('django.db.models.fields.DateTimeField', [], {'default': 'datetime.datetime(2014, 3, 8, 0, 0)'}),
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'modified': ('django.db.models.fields.DateTimeField', [], {'default': 'datetime.datetime(2014, 3, 8, 0, 0)'}),
            'tags': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'stream'", 'to': "orm['factopy.TagManager']"})
        },
        'factopy.tagmanager': {
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
            'remotename': ('django.db.models.fields.TextField', [], {'null': 'True'}),
            'size': ('django.db.models.fields.IntegerField', [], {'null': 'True'})
        },
        'requester.ftpserveraccount': {
            'Meta': {'object_name': 'FTPServerAccount', '_ormbases': ['requester.ServerAccount']},
            'hostname': ('django.db.models.fields.TextField', [], {}),
            u'serveraccount_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['requester.ServerAccount']", 'unique': 'True', 'primary_key': 'True'})
        },
        'requester.goesrequest': {
            'Meta': {'object_name': 'GOESRequest', '_ormbases': ['requester.Job']},
            u'job_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['requester.Job']", 'unique': 'True', 'primary_key': 'True'})
        },
        'requester.image': {
            'Meta': {'object_name': 'Image', '_ormbases': ['requester.File']},
            u'file_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['requester.File']", 'unique': 'True', 'primary_key': 'True'})
        },
        'requester.job': {
            'Meta': {'object_name': 'Job', '_ormbases': ['factopy.Process']},
            u'process_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['factopy.Process']", 'unique': 'True', 'primary_key': 'True'})
        },
        'requester.noaaadapt': {
            'Meta': {'object_name': 'NOAAAdapt', '_ormbases': ['factopy.Adapt']},
            u'adapt_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['factopy.Adapt']", 'unique': 'True', 'primary_key': 'True'}),
            'area': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['requester.Area']"}),
            'begin': ('django.db.models.fields.DateTimeField', [], {'null': 'True'}),
            'channels': ('django.db.models.fields.related.ManyToManyField', [], {'to': "orm['requester.Channel']", 'symmetrical': 'False'}),
            'created': ('django.db.models.fields.DateTimeField', [], {'auto_now_add': 'True', 'blank': 'True'}),
            'email_server': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['requester.EmailAccount']"}),
            'end': ('django.db.models.fields.DateTimeField', [], {'null': 'True'}),
            'max_simultaneous_request': ('django.db.models.fields.IntegerField', [], {}),
            'modified': ('django.db.models.fields.DateTimeField', [], {'auto_now': 'True', 'blank': 'True'}),
            'paused': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'request_server': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['requester.WebServerAccount']"}),
            'root_path': ('django.db.models.fields.TextField', [], {}),
            'title': ('django.db.models.fields.TextField', [], {'db_index': 'True'})
        },
        'requester.noaaemailchecker': {
            'Meta': {'object_name': 'NOAAEMailChecker', '_ormbases': ['requester.Job']},
            u'job_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['requester.Job']", 'unique': 'True', 'primary_key': 'True'}),
            'server': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['requester.EmailAccount']"})
        },
        'requester.noaaftpdownloader': {
            'Meta': {'object_name': 'NOAAFTPDownloader', '_ormbases': ['requester.Job']},
            'file': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['requester.File']", 'null': 'True'}),
            u'job_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['requester.Job']", 'unique': 'True', 'primary_key': 'True'}),
            'service': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['requester.FTPServerAccount']", 'null': 'True'})
        },
        'requester.qosmanager__impl': {
            'Meta': {'object_name': 'QOSManager__impl', '_ormbases': ['requester.Job']},
            u'job_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['requester.Job']", 'unique': 'True', 'primary_key': 'True'})
        },
        'requester.qosrequester': {
            'Meta': {'object_name': 'QOSRequester', '_ormbases': ['requester.Job']},
            u'job_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['requester.Job']", 'unique': 'True', 'primary_key': 'True'})
        },
        'requester.request': {
            'Meta': {'object_name': 'Request', '_ormbases': ['factopy.Material']},
            'adapt': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['requester.NOAAAdapt']"}),
            'aged': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
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
            'name': ('django.db.models.fields.TextField', [], {})
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

    complete_apps = ['requester']