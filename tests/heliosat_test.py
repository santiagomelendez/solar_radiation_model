import unittest
from netcdf import netcdf as nc
from datetime import datetime
from models import JobDescription
import os
import glob


class TestHeliosat(unittest.TestCase):

    def setUp(self):
        # os.system('rm -rf static.nc temporal_cache product')
        os.system('rm -rf temporal_cache product/estimated')
        os.system('rm -rf temporal_cache')
        os.system('cp -rf data mock_data')
        self.files = glob.glob('mock_data/goes13.*.BAND_01.nc')
        self.tile_cut = {
            "xc": [20, 30],
            "yc": [10, 15]
        }

    def tearDown(self):
        os.system('rm -rf mock_data')

    def verify_output(self):
        with nc.loader('tests/products/estimated/*.nc', self.tile_cut) as old:
            with nc.loader('products/estimated/*.nc', self.tile_cut) as new:
                valid = nc.getvar(old, 'globalradiation')
                max_vaild = valid[:].max()
                # It allow a 1% of the maximum value as the maximum error
                # threshold.
                threshold = max_vaild * 0.01
                calculated = nc.getvar(new, 'globalradiation')
                gtz = lambda m: m[calculated[:] >= 0]
                diff = gtz(calculated[:] - valid[:])
                print 'thr: ', threshold
                print 'min: ', gtz(calculated[:]).min(), '(', gtz(valid[:]).min(), ')'
                print 'max: ', gtz(calculated[:]).max(), '(', gtz(valid[:]).max(), ')'
                self.assertTrue((diff < threshold).all())
                shape = valid.shape
        return shape

    def test_main(self):
        config = {
            'algorithm': 'heliosat',
            'data': 'mock_data/goes13.2015.*.BAND_01.nc',
            'temporal_cache': 'temporal_cache',
            'product': 'products/estimated',
            'tile_cut': self.tile_cut
        }
        job = JobDescription(**config)
        begin = datetime.now()
        job.run()
        # heliosat.workwith(**config)
        end = datetime.now()
        shape = self.verify_output()
        elapsed = (end - begin).total_seconds()
        image_ratio = (30. * 14 * 2 / shape[0])
        scale_shapes = (2245. / shape[1]) * (3515. / shape[2]) * (image_ratio)
        estimated = elapsed * scale_shapes / 3600.
        print "Scaling total time to %.2f hours." % estimated
        print "Efficiency achieved: %.2f%%" % (3.5 / estimated * 100.)


if __name__ == '__main__':
    unittest.run()
