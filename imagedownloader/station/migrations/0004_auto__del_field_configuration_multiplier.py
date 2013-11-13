# -*- coding: utf-8 -*-
import datetime
from south.db import db
from south.v2 import SchemaMigration
from django.db import models


class Migration(SchemaMigration):

    def forwards(self, orm):
        # Deleting field 'Configuration.multiplier'
        db.delete_column('station_configuration', 'multiplier')


    def backwards(self, orm):

        # User chose to not deal with backwards NULL issues for 'Configuration.multiplier'
        raise RuntimeError("Cannot reverse this migration. 'Configuration.multiplier' and its values cannot be restored.")

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
            'between': ('django.db.models.fields.IntegerField', [], {'default': '0'}),
            'configuration': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['station.Configuration']"}),
            'finish': ('django.db.models.fields.DateTimeField', [], {'default': 'datetime.datetime(2013, 11, 12, 0, 0)'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'mean': ('django.db.models.fields.DecimalField', [], {'default': "'0'", 'max_digits': '5', 'decimal_places': '2'}),
            'refresh_presision': ('django.db.models.fields.IntegerField', [], {'default': '0'})
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