import unittest
from netcdf import netcdf as nc
from datetime import datetime
from models import JobDescription
import os
import glob
import tempfile


class TestHeliosat(unittest.TestCase):

    def setUp(self):
        tmp = tempfile.gettempdir() + '/temporal_cache'
        print tmp
        os.system('rm -rf static.nc product %s' % tmp)
        os.system('cp -rf data mock_data')
        self.files = glob.glob('mock_data/goes13.*.BAND_01.nc')[:-1]
        self.tile_cut = {
            "xc": [20, 30],
            "yc": [10, 15]
        }

    def tearDown(self):
        os.system('rm -rf mock_data')

    def translate_file(self, path, filename):
        return '%s/%s' % (path, filename.split('/')[-1])

    def verify_output(self, files):
        tested = map(lambda f:
                     self.translate_file('tests/products/estimated', f),
                     files)
        products = map(lambda f: self.translate_file('products/estimated', f),
                       files)
        with nc.loader(tested, self.tile_cut) as old:
            with nc.loader(products, self.tile_cut) as new:
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
            'static_file': 'static.nc',
            'data': self.files,
            'temporal_cache': None,
            'product': 'products/estimated',
            'tile_cut': self.tile_cut,
            'hard': 'gpu',
        }
        job = JobDescription(**config)
        self.files = job.filter_data(self.files)
        begin = datetime.now()
        job.run()
        end = datetime.now()
        shape = self.verify_output(self.files)
        elapsed = (end - begin).total_seconds()
        image_ratio = (30. * 14 * 2 / shape[0])
        scale_shapes = (2245. / shape[1]) * (3515. / shape[2]) * (image_ratio)
        estimated = elapsed * scale_shapes / 3600.
        print "Scaling total time to %.2f hours." % estimated
        print "Efficiency achieved: %.2f%%" % (3.5 / estimated * 100.)


if __name__ == '__main__':
    unittest.run()
