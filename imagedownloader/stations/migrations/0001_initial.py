# -*- coding: utf-8 -*-
import datetime
from south.db import db
from south.v2 import SchemaMigration
from django.db import models


class Migration(SchemaMigration):

    def forwards(self, orm):
        # Adding model 'OpticFilter'
        db.create_table('stations_opticfilter', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('name', self.gf('django.db.models.fields.TextField')(db_index=True)),
        ))
        db.send_create_signal('stations', ['OpticFilter'])

        # Adding model 'Brand'
        db.create_table('stations_brand', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('name', self.gf('django.db.models.fields.TextField')(db_index=True)),
        ))
        db.send_create_signal('stations', ['Brand'])

        # Adding model 'Product'
        db.create_table('stations_product', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('brand', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['stations.Brand'])),
            ('name', self.gf('django.db.models.fields.TextField')(db_index=True)),
            ('specifications', self.gf('django.db.models.fields.TextField')(db_index=True)),
        ))
        db.send_create_signal('stations', ['Product'])

        # Adding model 'Device'
        db.create_table('stations_device', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('polymorphic_ctype', self.gf('django.db.models.fields.related.ForeignKey')(related_name='polymorphic_stations.device_set', null=True, to=orm['contenttypes.ContentType'])),
            ('product', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['stations.Product'])),
            ('serial_number', self.gf('django.db.models.fields.TextField')(default='', db_index=True)),
            ('description', self.gf('django.db.models.fields.TextField')(default='', db_index=True)),
        ))
        db.send_create_signal('stations', ['Device'])

        # Adding model 'Sensor'
        db.create_table('stations_sensor', (
            ('device_ptr', self.gf('django.db.models.fields.related.OneToOneField')(to=orm['stations.Device'], unique=True, primary_key=True)),
            ('optic_filter', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['stations.OpticFilter'], null=True)),
        ))
        db.send_create_signal('stations', ['Sensor'])

        # Adding model 'Datalogger'
        db.create_table('stations_datalogger', (
            ('device_ptr', self.gf('django.db.models.fields.related.OneToOneField')(to=orm['stations.Device'], unique=True, primary_key=True)),
        ))
        db.send_create_signal('stations', ['Datalogger'])

        # Adding model 'Tracker'
        db.create_table('stations_tracker', (
            ('device_ptr', self.gf('django.db.models.fields.related.OneToOneField')(to=orm['stations.Device'], unique=True, primary_key=True)),
        ))
        db.send_create_signal('stations', ['Tracker'])

        # Adding model 'ShadowBall'
        db.create_table('stations_shadowball', (
            ('device_ptr', self.gf('django.db.models.fields.related.OneToOneField')(to=orm['stations.Device'], unique=True, primary_key=True)),
        ))
        db.send_create_signal('stations', ['ShadowBall'])

        # Adding model 'InclinedSupport'
        db.create_table('stations_inclinedsupport', (
            ('device_ptr', self.gf('django.db.models.fields.related.OneToOneField')(to=orm['stations.Device'], unique=True, primary_key=True)),
            ('angle', self.gf('django.db.models.fields.DecimalField')(default='0.00', max_digits=7, decimal_places=4)),
        ))
        db.send_create_signal('stations', ['InclinedSupport'])

        # Adding model 'SensorCalibration'
        db.create_table('stations_sensorcalibration', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('sensor', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['stations.Sensor'])),
            ('coefficient', self.gf('django.db.models.fields.DecimalField')(default='0.00', max_digits=10, decimal_places=7)),
            ('shift', self.gf('django.db.models.fields.DecimalField')(default='0.00', max_digits=10, decimal_places=7)),
        ))
        db.send_create_signal('stations', ['SensorCalibration'])

        # Adding model 'Position'
        db.create_table('stations_position', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('station', self.gf('django.db.models.fields.related.ForeignKey')(default=None, to=orm['stations.Station'], null=True)),
            ('latitude', self.gf('django.db.models.fields.DecimalField')(default='0.00', max_digits=10, decimal_places=7)),
            ('longitude', self.gf('django.db.models.fields.DecimalField')(default='0.00', max_digits=10, decimal_places=7)),
        ))
        db.send_create_signal('stations', ['Position'])

        # Adding model 'Station'
        db.create_table('stations_station', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('name', self.gf('django.db.models.fields.TextField')(db_index=True)),
        ))
        db.send_create_signal('stations', ['Station'])

        # Adding model 'Configuration'
        db.create_table('stations_configuration', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('begin', self.gf('django.db.models.fields.DateTimeField')(default=datetime.datetime(2014, 1, 17, 0, 0))),
            ('end', self.gf('django.db.models.fields.DateTimeField')(null=True, blank=True)),
            ('position', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['stations.Position'])),
            ('calibration', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['stations.SensorCalibration'])),
            ('created', self.gf('django.db.models.fields.DateTimeField')(default=datetime.datetime(2014, 1, 17, 0, 0))),
            ('modified', self.gf('django.db.models.fields.DateTimeField')(default=datetime.datetime(2014, 1, 17, 0, 0))),
            ('backup', self.gf('django.db.models.fields.TextField')(default='')),
        ))
        db.send_create_signal('stations', ['Configuration'])

        # Adding M2M table for field devices on 'Configuration'
        db.create_table('stations_configuration_devices', (
            ('id', models.AutoField(verbose_name='ID', primary_key=True, auto_created=True)),
            ('configuration', models.ForeignKey(orm['stations.configuration'], null=False)),
            ('device', models.ForeignKey(orm['stations.device'], null=False))
        ))
        db.create_unique('stations_configuration_devices', ['configuration_id', 'device_id'])

        # Adding model 'Measurement'
        db.create_table('stations_measurement', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('mean', self.gf('django.db.models.fields.DecimalField')(default='0.00', max_digits=5, decimal_places=2)),
            ('between', self.gf('django.db.models.fields.IntegerField')(default=0)),
            ('finish', self.gf('django.db.models.fields.DateTimeField')(default=datetime.datetime(2014, 1, 17, 0, 0))),
            ('refresh_presision', self.gf('django.db.models.fields.IntegerField')(default=0)),
            ('configuration', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['stations.Configuration'])),
        ))
        db.send_create_signal('stations', ['Measurement'])

        # Adding unique constraint on 'Measurement', fields ['configuration', 'finish']
        db.create_unique('stations_measurement', ['configuration_id', 'finish'])


    def backwards(self, orm):
        # Removing unique constraint on 'Measurement', fields ['configuration', 'finish']
        db.delete_unique('stations_measurement', ['configuration_id', 'finish'])

        # Deleting model 'OpticFilter'
        db.delete_table('stations_opticfilter')

        # Deleting model 'Brand'
        db.delete_table('stations_brand')

        # Deleting model 'Product'
        db.delete_table('stations_product')

        # Deleting model 'Device'
        db.delete_table('stations_device')

        # Deleting model 'Sensor'
        db.delete_table('stations_sensor')

        # Deleting model 'Datalogger'
        db.delete_table('stations_datalogger')

        # Deleting model 'Tracker'
        db.delete_table('stations_tracker')

        # Deleting model 'ShadowBall'
        db.delete_table('stations_shadowball')

        # Deleting model 'InclinedSupport'
        db.delete_table('stations_inclinedsupport')

        # Deleting model 'SensorCalibration'
        db.delete_table('stations_sensorcalibration')

        # Deleting model 'Position'
        db.delete_table('stations_position')

        # Deleting model 'Station'
        db.delete_table('stations_station')

        # Deleting model 'Configuration'
        db.delete_table('stations_configuration')

        # Removing M2M table for field devices on 'Configuration'
        db.delete_table('stations_configuration_devices')

        # Deleting model 'Measurement'
        db.delete_table('stations_measurement')


    models = {
        'contenttypes.contenttype': {
            'Meta': {'ordering': "('name',)", 'unique_together': "(('app_label', 'model'),)", 'object_name': 'ContentType', 'db_table': "'django_content_type'"},
            'app_label': ('django.db.models.fields.CharField', [], {'max_length': '100'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'model': ('django.db.models.fields.CharField', [], {'max_length': '100'}),
            'name': ('django.db.models.fields.CharField', [], {'max_length': '100'})
        },
        'stations.brand': {
            'Meta': {'object_name': 'Brand'},
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'name': ('django.db.models.fields.TextField', [], {'db_index': 'True'})
        },
        'stations.configuration': {
            'Meta': {'object_name': 'Configuration'},
            'backup': ('django.db.models.fields.TextField', [], {'default': "''"}),
            'begin': ('django.db.models.fields.DateTimeField', [], {'default': 'datetime.datetime(2014, 1, 17, 0, 0)'}),
            'calibration': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['stations.SensorCalibration']"}),
            'created': ('django.db.models.fields.DateTimeField', [], {'default': 'datetime.datetime(2014, 1, 17, 0, 0)'}),
            'devices': ('django.db.models.fields.related.ManyToManyField', [], {'related_name': "'configurations'", 'symmetrical': 'False', 'to': "orm['stations.Device']"}),
            'end': ('django.db.models.fields.DateTimeField', [], {'null': 'True', 'blank': 'True'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'modified': ('django.db.models.fields.DateTimeField', [], {'default': 'datetime.datetime(2014, 1, 17, 0, 0)'}),
            'position': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['stations.Position']"})
        },
        'stations.datalogger': {
            'Meta': {'object_name': 'Datalogger', '_ormbases': ['stations.Device']},
            'device_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['stations.Device']", 'unique': 'True', 'primary_key': 'True'})
        },
        'stations.device': {
            'Meta': {'object_name': 'Device'},
            'description': ('django.db.models.fields.TextField', [], {'default': "''", 'db_index': 'True'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'polymorphic_ctype': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'polymorphic_stations.device_set'", 'null': 'True', 'to': "orm['contenttypes.ContentType']"}),
            'product': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['stations.Product']"}),
            'serial_number': ('django.db.models.fields.TextField', [], {'default': "''", 'db_index': 'True'})
        },
        'stations.inclinedsupport': {
            'Meta': {'object_name': 'InclinedSupport', '_ormbases': ['stations.Device']},
            'angle': ('django.db.models.fields.DecimalField', [], {'default': "'0.00'", 'max_digits': '7', 'decimal_places': '4'}),
            'device_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['stations.Device']", 'unique': 'True', 'primary_key': 'True'})
        },
        'stations.measurement': {
            'Meta': {'unique_together': "(('configuration', 'finish'),)", 'object_name': 'Measurement'},
            'between': ('django.db.models.fields.IntegerField', [], {'default': '0'}),
            'configuration': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['stations.Configuration']"}),
            'finish': ('django.db.models.fields.DateTimeField', [], {'default': 'datetime.datetime(2014, 1, 17, 0, 0)'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'mean': ('django.db.models.fields.DecimalField', [], {'default': "'0.00'", 'max_digits': '5', 'decimal_places': '2'}),
            'refresh_presision': ('django.db.models.fields.IntegerField', [], {'default': '0'})
        },
        'stations.opticfilter': {
            'Meta': {'object_name': 'OpticFilter'},
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'name': ('django.db.models.fields.TextField', [], {'db_index': 'True'})
        },
        'stations.position': {
            'Meta': {'object_name': 'Position'},
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'latitude': ('django.db.models.fields.DecimalField', [], {'default': "'0.00'", 'max_digits': '10', 'decimal_places': '7'}),
            'longitude': ('django.db.models.fields.DecimalField', [], {'default': "'0.00'", 'max_digits': '10', 'decimal_places': '7'}),
            'station': ('django.db.models.fields.related.ForeignKey', [], {'default': 'None', 'to': "orm['stations.Station']", 'null': 'True'})
        },
        'stations.product': {
            'Meta': {'object_name': 'Product'},
            'brand': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['stations.Brand']"}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'name': ('django.db.models.fields.TextField', [], {'db_index': 'True'}),
            'specifications': ('django.db.models.fields.TextField', [], {'db_index': 'True'})
        },
        'stations.sensor': {
            'Meta': {'object_name': 'Sensor', '_ormbases': ['stations.Device']},
            'device_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['stations.Device']", 'unique': 'True', 'primary_key': 'True'}),
            'optic_filter': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['stations.OpticFilter']", 'null': 'True'})
        },
        'stations.sensorcalibration': {
            'Meta': {'object_name': 'SensorCalibration'},
            'coefficient': ('django.db.models.fields.DecimalField', [], {'default': "'0.00'", 'max_digits': '10', 'decimal_places': '7'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'sensor': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['stations.Sensor']"}),
            'shift': ('django.db.models.fields.DecimalField', [], {'default': "'0.00'", 'max_digits': '10', 'decimal_places': '7'})
        },
        'stations.shadowball': {
            'Meta': {'object_name': 'ShadowBall', '_ormbases': ['stations.Device']},
            'device_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['stations.Device']", 'unique': 'True', 'primary_key': 'True'})
        },
        'stations.station': {
            'Meta': {'object_name': 'Station'},
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'name': ('django.db.models.fields.TextField', [], {'db_index': 'True'})
        },
        'stations.tracker': {
            'Meta': {'object_name': 'Tracker', '_ormbases': ['stations.Device']},
            'device_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['stations.Device']", 'unique': 'True', 'primary_key': 'True'})
        }
    }

    complete_apps = ['stations']