import unittest
from models import heliosat
import numpy as np
from netcdf import netcdf as nc
from datetime import datetime
import os
import glob


class TestHeliosat(unittest.TestCase):

    def setUp(self):
        os.system('rm -rf static.nc temporal_cache products/estimated')
        os.system('cp -rf data mock_data')
        self.files = glob.glob('mock_data/goes13.*.BAND_01.nc')

    def tearDown(self):
        os.system('rm -rf mock_data')

    def test_main(self):
        begin = datetime.now()
        heliosat.workwith('mock_data/goes13.2015.*.BAND_01.nc')
        end = datetime.now()
        elapsed = (end - begin).total_seconds()
        first, last = min(self.files), max(self.files)
        to_dt = heliosat.to_datetime
        processed = (to_dt(last) - to_dt(first)).total_seconds()
        processed_days = processed / 3600. / 24
        scale_shapes = (2245. / 86) * (3515. / 180) * (30. / processed_days)
        estimated = elapsed * scale_shapes / 3600.
        print "Scaling total time to %.2f hours." % estimated
        print "Efficiency achieved: %.2f%%" % (3.5 / estimated * 100.)


if __name__ == '__main__':
    unittest.run()
