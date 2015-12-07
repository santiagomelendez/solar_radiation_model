from __future__ import print_function
import unittest
from netcdf import netcdf as nc
from models import JobDescription
from models.cache import Cache, StaticCache
import os
import glob


class TestHeliosat(unittest.TestCase):

    def setUp(self):
        os.system('rm -rf static.nc product')
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

    def verify_output(self, files, output, config):
        tested = map(lambda f:
                     self.translate_file('tests/products/estimated', f),
                     files)
        with nc.loader(tested, self.tile_cut) as old:
            valid = nc.getvar(old, 'globalradiation')
            max_vaild = valid[:].max()
            # It allow a 1% of the maximum value as the maximum error
            # threshold.
            threshold = max_vaild * 0.01
            calculated = output.globalradiation
            if config['product']:
                products = map(lambda f:
                               self.translate_file('products/estimated', f),
                               files)
                with nc.loader(products, self.tile_cut) as new:
                    calculated = nc.getvar(new, 'globalradiation')
            gtz = lambda m: m[calculated[:] >= 0]
            diff = gtz(calculated[:] - valid[:])
            print('thr: {:}'.format(threshold))
            print('min: {:} ({:})'.format(gtz(calculated[:]).min(),
                                          gtz(valid[:]).min()))
            print('max: {:} ({:})'.format(gtz(calculated[:]).max(),
                                          gtz(valid[:]).max()))
            self.assertTrue((diff < threshold).all())
            shape = valid.shape
        return shape

    def test_main(self):
        config = {
            'algorithm': 'heliosat',
            'static_file': 'static.nc',
            'data': self.files,
            'product': None,  # 'products/estimated',
            'tile_cut': self.tile_cut,
            'hard': 'gpu',
        }
        job = JobDescription(**config)
        self.files = job.filter_data(self.files)
        intern_elapsed, output = job.run()
        shape = self.verify_output(self.files, output, config)
        image_ratio = (15. * 12. * 2. / shape[0])
        scale_shapes = (2260. / shape[1]) * (4360. / shape[2]) * (image_ratio)
        cores = 24. * 7.
        intern_estimated = intern_elapsed * (scale_shapes / cores) / 3600.
        print("Scaling intern time to {:.2f} hours.".format(intern_estimated))
        print("Needed efficiency achieved: {:.2f}%".format(
            0.5 / intern_estimated * 100.))

    def test_with_loaded_files(self):
        files = JobDescription.filter_data(self.files)
        config = {
            'algorithm': 'heliosat',
            'static_file': StaticCache('static.nc', files, self.tile_cut),
            'data': Cache(files, tile_cut=self.tile_cut),
            'product': None,
            'tile_cut': self.tile_cut,
            'hard': 'gpu',
        }
        job = JobDescription(**config)
        intern_elapsed, output = job.run()
        shape = self.verify_output(files, output, config)
        image_ratio = (15. * 12. * 2. / shape[0])
        scale_shapes = (2260. / shape[1]) * (4360. / shape[2]) * (image_ratio)
        cores = 24. * 7.
        intern_estimated = intern_elapsed * (scale_shapes / cores) / 3600.
        print("Scaling intern time to {:.2f} hours.".format(intern_estimated))
        print("Needed efficiency achieved: {:.2f}%".format(
            0.5 / intern_estimated * 100.))


if __name__ == '__main__':
    unittest.run()
