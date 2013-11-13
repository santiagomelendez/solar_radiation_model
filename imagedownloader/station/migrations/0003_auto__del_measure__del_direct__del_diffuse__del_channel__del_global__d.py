# -*- coding: utf-8 -*-
import datetime
from south.db import db
from south.v2 import SchemaMigration
from django.db import models


class Migration(SchemaMigration):

    def forwards(self, orm):
        # Deleting model 'Measure'
        db.delete_table('station_measure')

        # Deleting model 'Direct'
        db.delete_table('station_direct')

        # Deleting model 'Diffuse'
        db.delete_table('station_diffuse')

        # Deleting model 'Channel'
        db.delete_table('station_channel')

        # Deleting model 'Global'
        db.delete_table('station_global')

        # Deleting model 'StaticConfiguration'
        db.delete_table('station_staticconfiguration')

        # Adding model 'Device'
        db.create_table('station_device', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('serial_number', self.gf('django.db.models.fields.TextField')(default='', db_index=True)),
            ('product', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['station.Product'])),
            ('description', self.gf('django.db.models.fields.TextField')(db_index=True)),
        ))
        db.send_create_signal('station', ['Device'])

        # Adding model 'Tracker'
        db.create_table('station_tracker', (
            ('device_ptr', self.gf('django.db.models.fields.related.OneToOneField')(to=orm['station.Device'], unique=True, primary_key=True)),
        ))
        db.send_create_signal('station', ['Tracker'])

        # Adding model 'SensorCalibration'
        db.create_table('station_sensorcalibration', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('coefficient', self.gf('django.db.models.fields.DecimalField')(default='0', max_digits=10, decimal_places=7)),
            ('shift', self.gf('django.db.models.fields.DecimalField')(default='0', max_digits=10, decimal_places=7)),
        ))
        db.send_create_signal('station', ['SensorCalibration'])

        # Adding model 'InclinedSupport'
        db.create_table('station_inclinedsupport', (
            ('device_ptr', self.gf('django.db.models.fields.related.OneToOneField')(to=orm['station.Device'], unique=True, primary_key=True)),
            ('angle', self.gf('django.db.models.fields.DecimalField')(default='0', max_digits=7, decimal_places=4)),
        ))
        db.send_create_signal('station', ['InclinedSupport'])

        # Adding model 'Measurement'
        db.create_table('station_measurement', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('mean', self.gf('django.db.models.fields.DecimalField')(max_digits=5, decimal_places=2)),
            ('between', self.gf('django.db.models.fields.IntegerField')()),
            ('finish', self.gf('django.db.models.fields.DateTimeField')()),
            ('refresh_presision', self.gf('django.db.models.fields.IntegerField')()),
            ('configuration', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['station.Configuration'])),
        ))
        db.send_create_signal('station', ['Measurement'])

        # Adding model 'ShadowBall'
        db.create_table('station_shadowball', (
            ('device_ptr', self.gf('django.db.models.fields.related.OneToOneField')(to=orm['station.Device'], unique=True, primary_key=True)),
        ))
        db.send_create_signal('station', ['ShadowBall'])

        # Deleting field 'Station.position'
        db.delete_column('station_station', 'position_id')

        # Adding field 'Position.station'
        db.add_column('station_position', 'station',
                      self.gf('django.db.models.fields.related.ForeignKey')(default=None, to=orm['station.Station'], null=True),
                      keep_default=False)

        # Deleting field 'Datalogger.id'
        db.delete_column('station_datalogger', 'id')

        # Deleting field 'Datalogger.label'
        db.delete_column('station_datalogger', 'label')

        # Adding field 'Datalogger.device_ptr'
        db.add_column('station_datalogger', 'device_ptr',
                      self.gf('django.db.models.fields.related.OneToOneField')(default=None, to=orm['station.Device'], unique=True, primary_key=True),
                      keep_default=False)

        # Deleting field 'Sensor.serial_number'
        db.delete_column('station_sensor', 'serial_number')

        # Deleting field 'Sensor.product'
        db.delete_column('station_sensor', 'product_id')

        # Deleting field 'Sensor.id'
        db.delete_column('station_sensor', 'id')

        # Adding field 'Sensor.device_ptr'
        db.add_column('station_sensor', 'device_ptr',
                      self.gf('django.db.models.fields.related.OneToOneField')(default=None, to=orm['station.Device'], unique=True, primary_key=True),
                      keep_default=False)

        # Deleting field 'Configuration.frequency'
        db.delete_column('station_configuration', 'frequency')

        # Deleting field 'Configuration.frequency_save'
        db.delete_column('station_configuration', 'frequency_save')

        # Deleting field 'Configuration.calibration_value'
        db.delete_column('station_configuration', 'calibration_value')

        # Deleting field 'Configuration.sensor'
        db.delete_column('station_configuration', 'sensor_id')

        # Deleting field 'Configuration.datetime'
        db.delete_column('station_configuration', 'datetime')

        # Adding field 'Configuration.calibration'
        db.add_column('station_configuration', 'calibration',
                      self.gf('django.db.models.fields.related.ForeignKey')(default=None, to=orm['station.SensorCalibration']),
                      keep_default=False)

        # Adding field 'Configuration.created'
        db.add_column('station_configuration', 'created',
                      self.gf('django.db.models.fields.DateTimeField')(default=datetime.datetime(2013, 11, 12, 0, 0)),
                      keep_default=False)

        # Adding field 'Configuration.modified'
        db.add_column('station_configuration', 'modified',
                      self.gf('django.db.models.fields.DateTimeField')(default=datetime.datetime(2013, 11, 12, 0, 0)),
                      keep_default=False)

        # Adding M2M table for field devices on 'Configuration'
        db.create_table('station_configuration_devices', (
            ('id', models.AutoField(verbose_name='ID', primary_key=True, auto_created=True)),
            ('configuration', models.ForeignKey(orm['station.configuration'], null=False)),
            ('device', models.ForeignKey(orm['station.device'], null=False))
        ))
        db.create_unique('station_configuration_devices', ['configuration_id', 'device_id'])

        # Deleting field 'OpticFilter.description'
        db.delete_column('station_opticfilter', 'description')


    def backwards(self, orm):
        # Adding model 'Measure'
        db.create_table('station_measure', (
            ('date', self.gf('django.db.models.fields.DateTimeField')()),
            ('value', self.gf('django.db.models.fields.DecimalField')(max_digits=5, decimal_places=2)),
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('configuration', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['station.Configuration'])),
        ))
        db.send_create_signal('station', ['Measure'])

        # Adding model 'Direct'
        db.create_table('station_direct', (
            ('configuration_ptr', self.gf('django.db.models.fields.related.OneToOneField')(to=orm['station.Configuration'], unique=True, primary_key=True)),
        ))
        db.send_create_signal('station', ['Direct'])

        # Adding model 'Diffuse'
        db.create_table('station_diffuse', (
            ('staticconfiguration_ptr', self.gf('django.db.models.fields.related.OneToOneField')(to=orm['station.StaticConfiguration'], unique=True, primary_key=True)),
            ('shadow_ball', self.gf('django.db.models.fields.BooleanField')(default=False)),
        ))
        db.send_create_signal('station', ['Diffuse'])

        # Adding model 'Channel'
        db.create_table('station_channel', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('datalogger', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['station.Datalogger'])),
            ('name', self.gf('django.db.models.fields.TextField')(db_index=True)),
        ))
        db.send_create_signal('station', ['Channel'])

        # Adding model 'Global'
        db.create_table('station_global', (
            ('staticconfiguration_ptr', self.gf('django.db.models.fields.related.OneToOneField')(to=orm['station.StaticConfiguration'], unique=True, primary_key=True)),
        ))
        db.send_create_signal('station', ['Global'])

        # Adding model 'StaticConfiguration'
        db.create_table('station_staticconfiguration', (
            ('configuration_ptr', self.gf('django.db.models.fields.related.OneToOneField')(to=orm['station.Configuration'], unique=True, primary_key=True)),
            ('angle', self.gf('django.db.models.fields.DecimalField')(max_digits=4, decimal_places=2)),
        ))
        db.send_create_signal('station', ['StaticConfiguration'])

        # Deleting model 'Device'
        db.delete_table('station_device')

        # Deleting model 'Tracker'
        db.delete_table('station_tracker')

        # Deleting model 'SensorCalibration'
        db.delete_table('station_sensorcalibration')

        # Deleting model 'InclinedSupport'
        db.delete_table('station_inclinedsupport')

        # Deleting model 'Measurement'
        db.delete_table('station_measurement')

        # Deleting model 'ShadowBall'
        db.delete_table('station_shadowball')


        # User chose to not deal with backwards NULL issues for 'Station.position'
        raise RuntimeError("Cannot reverse this migration. 'Station.position' and its values cannot be restored.")
        # Deleting field 'Position.station'
        db.delete_column('station_position', 'station_id')


        # User chose to not deal with backwards NULL issues for 'Datalogger.id'
        raise RuntimeError("Cannot reverse this migration. 'Datalogger.id' and its values cannot be restored.")

        # User chose to not deal with backwards NULL issues for 'Datalogger.label'
        raise RuntimeError("Cannot reverse this migration. 'Datalogger.label' and its values cannot be restored.")
        # Deleting field 'Datalogger.device_ptr'
        db.delete_column('station_datalogger', 'device_ptr_id')


        # User chose to not deal with backwards NULL issues for 'Sensor.serial_number'
        raise RuntimeError("Cannot reverse this migration. 'Sensor.serial_number' and its values cannot be restored.")

        # User chose to not deal with backwards NULL issues for 'Sensor.product'
        raise RuntimeError("Cannot reverse this migration. 'Sensor.product' and its values cannot be restored.")

        # User chose to not deal with backwards NULL issues for 'Sensor.id'
        raise RuntimeError("Cannot reverse this migration. 'Sensor.id' and its values cannot be restored.")
        # Deleting field 'Sensor.device_ptr'
        db.delete_column('station_sensor', 'device_ptr_id')


        # User chose to not deal with backwards NULL issues for 'Configuration.frequency'
        raise RuntimeError("Cannot reverse this migration. 'Configuration.frequency' and its values cannot be restored.")

        # User chose to not deal with backwards NULL issues for 'Configuration.frequency_save'
        raise RuntimeError("Cannot reverse this migration. 'Configuration.frequency_save' and its values cannot be restored.")

        # User chose to not deal with backwards NULL issues for 'Configuration.calibration_value'
        raise RuntimeError("Cannot reverse this migration. 'Configuration.calibration_value' and its values cannot be restored.")

        # User chose to not deal with backwards NULL issues for 'Configuration.sensor'
        raise RuntimeError("Cannot reverse this migration. 'Configuration.sensor' and its values cannot be restored.")

        # User chose to not deal with backwards NULL issues for 'Configuration.datetime'
        raise RuntimeError("Cannot reverse this migration. 'Configuration.datetime' and its values cannot be restored.")
        # Deleting field 'Configuration.calibration'
        db.delete_column('station_configuration', 'calibration_id')

        # Deleting field 'Configuration.created'
        db.delete_column('station_configuration', 'created')

        # Deleting field 'Configuration.modified'
        db.delete_column('station_configuration', 'modified')

        # Removing M2M table for field devices on 'Configuration'
        db.delete_table('station_configuration_devices')


        # User chose to not deal with backwards NULL issues for 'OpticFilter.description'
        raise RuntimeError("Cannot reverse this migration. 'OpticFilter.description' and its values cannot be restored.")

    models = {
        'station.brand': {
            'Meta': {'object_name': 'Brand'},
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'name': ('django.db.models.fields.TextField', [], {'db_index': 'True'})
        },
        'station.configuration': {
            'Meta': {'object_name': 'Configuration'},
            'calibration': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['station.SensorCalibration']"}),
            'created': ('django.db.models.fields.DateTimeField', [], {'default': 'datetime.datetime(2013, 11, 12, 0, 0)'}),
            'devices': ('django.db.models.fields.related.ManyToManyField', [], {'related_name': "'configurations'", 'symmetrical': 'False', 'to': "orm['station.Device']"}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'modified': ('django.db.models.fields.DateTimeField', [], {'default': 'datetime.datetime(2013, 11, 12, 0, 0)'}),
            'multiplier': ('django.db.models.fields.DecimalField', [], {'max_digits': '5', 'decimal_places': '2'}),
            'position': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['station.Position']"})
        },
        'station.datalogger': {
            'Meta': {'object_name': 'Datalogger', '_ormbases': ['station.Device']},
            'device_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['station.Device']", 'unique': 'True', 'primary_key': 'True'})
        },
        'station.device': {
            'Meta': {'object_name': 'Device'},
            'description': ('django.db.models.fields.TextField', [], {'db_index': 'True'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'product': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['station.Product']"}),
            'serial_number': ('django.db.models.fields.TextField', [], {'default': "''", 'db_index': 'True'})
        },
        'station.inclinedsupport': {
            'Meta': {'object_name': 'InclinedSupport', '_ormbases': ['station.Device']},
            'angle': ('django.db.models.fields.DecimalField', [], {'default': "'0'", 'max_digits': '7', 'decimal_places': '4'}),
            'device_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['station.Device']", 'unique': 'True', 'primary_key': 'True'})
        },
        'station.measurement': {
            'Meta': {'object_name': 'Measurement'},
            'between': ('django.db.models.fields.IntegerField', [], {}),
            'configuration': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['station.Configuration']"}),
            'finish': ('django.db.models.fields.DateTimeField', [], {}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'mean': ('django.db.models.fields.DecimalField', [], {'max_digits': '5', 'decimal_places': '2'}),
            'refresh_presision': ('django.db.models.fields.IntegerField', [], {})
        },
        'station.opticfilter': {
            'Meta': {'object_name': 'OpticFilter'},
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'name': ('django.db.models.fields.TextField', [], {'db_index': 'True'})
        },
        'station.position': {
            'Meta': {'object_name': 'Position'},
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'latitude': ('django.db.models.fields.DecimalField', [], {'default': "'0'", 'max_digits': '10', 'decimal_places': '7'}),
            'longitude': ('django.db.models.fields.DecimalField', [], {'default': "'0'", 'max_digits': '10', 'decimal_places': '7'}),
            'station': ('django.db.models.fields.related.ForeignKey', [], {'default': 'None', 'to': "orm['station.Station']", 'null': 'True'})
        },
        'station.product': {
            'Meta': {'object_name': 'Product'},
            'brand': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['station.Brand']"}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'name': ('django.db.models.fields.TextField', [], {'db_index': 'True'})
        },
        'station.sensor': {
            'Meta': {'object_name': 'Sensor', '_ormbases': ['station.Device']},
            'device_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['station.Device']", 'unique': 'True', 'primary_key': 'True'}),
            'optic_filter': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['station.OpticFilter']"})
        },
        'station.sensorcalibration': {
            'Meta': {'object_name': 'SensorCalibration'},
            'coefficient': ('django.db.models.fields.DecimalField', [], {'default': "'0'", 'max_digits': '10', 'decimal_places': '7'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'shift': ('django.db.models.fields.DecimalField', [], {'default': "'0'", 'max_digits': '10', 'decimal_places': '7'})
        },
        'station.shadowball': {
            'Meta': {'object_name': 'ShadowBall', '_ormbases': ['station.Device']},
            'device_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['station.Device']", 'unique': 'True', 'primary_key': 'True'})
        },
        'station.station': {
            'Meta': {'object_name': 'Station'},
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'name': ('django.db.models.fields.TextField', [], {'db_index': 'True'})
        },
        'station.tracker': {
            'Meta': {'object_name': 'Tracker', '_ormbases': ['station.Device']},
            'device_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['station.Device']", 'unique': 'True', 'primary_key': 'True'})
        }
    }

    complete_apps = ['station']