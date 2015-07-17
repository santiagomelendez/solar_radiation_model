import unittest
from models import heliosat
import numpy as np
from netcdf import netcdf as nc
from datetime import datetime
from models.cache import DIMS
from models import JobDescription, helpers
import os
import glob


class TestHeliosat(unittest.TestCase):

    def setUp(self):
        # os.system('rm -rf static.nc temporal_cache products')
        os.system('rm -rf temporal_cache products/estimated')
        os.system('cp -rf data mock_data')
        self.files = glob.glob('mock_data/goes13.*.BAND_01.nc')

    def tearDown(self):
        os.system('rm -rf mock_data')

    def verify_output(self):
        with nc.loader('tests/products/estimated/*.nc', DIMS) as old_root:
            with nc.loader('products/estimated/*.nc', DIMS) as new_root:
                valid = nc.getvar(old_root, 'globalradiation')
                max_vaild = valid[:].max()
                # It allow a 1% of the maximum value as the maximum error
                # threshold.
                threshold = max_vaild * 0.01
                calculated = nc.getvar(new_root, 'globalradiation')
                gtz = lambda m: m[calculated[:] >= 0]
                diff = gtz(calculated[:] - valid[:])
                print 'min: ', gtz(calculated[:]).min(), '(', gtz(valid[:]).min(), ')'
                print 'max: ', gtz(calculated[:]).max(), '(', gtz(valid[:]).max(), ')'
                self.assertTrue((diff < threshold).all())

    def test_main(self):
        config = {
            'algorithm': 'heliosat',
            'data': 'mock_data/goes13.2015.*.BAND_01.nc',
            'temporal_cache': 'temporal_cache',
            'product': 'products/estimated'
        }
        job = JobDescription(**config)
        begin = datetime.now()
        job.run()
        # heliosat.workwith(**config)
        end = datetime.now()
        self.verify_output()
        elapsed = (end - begin).total_seconds()
        first, last = min(self.files), max(self.files)
        to_dt = helpers.to_datetime
        processed = (to_dt(last) - to_dt(first)).total_seconds()
        processed_days = processed / 3600. / 24
        scale_shapes = (2245. / 86) * (3515. / 180) * (30. / processed_days)
        estimated = elapsed * scale_shapes / 3600.
        print "Scaling total time to %.2f hours." % estimated
        print "Efficiency achieved: %.2f%%" % (3.5 / estimated * 100.)


if __name__ == '__main__':
    unittest.run()
