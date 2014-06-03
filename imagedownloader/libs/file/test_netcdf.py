import unittest
from netCDF4 import Dataset
from libs.file import netcdf as nc
import os
import stat


class TestNetcdf(unittest.TestCase):

    def createRefFile(self, filename):
        ref = Dataset(
            filename,
            mode="w",
            clobber=True,
            format='NETCDF3_CLASSIC')
        ref.createDimension('auditCount', 2)
        ref.createDimension('auditSize', 80)
        ref.createDimension('xc', 200)
        ref.createDimension('yc', 100)
        ref.createDimension('time', 1)
        ref.createVariable(
            'time',
            'i4',
            dimensions=('time',),
            zlib=True,
            fill_value=0)
        ref.createVariable(
            'lat',
            'f4',
            dimensions=('yc', 'xc'),
            zlib=True,
            fill_value=0.0)
        ref.createVariable(
            'lon',
            'f4',
            dimensions=('yc', 'xc'),
            zlib=True,
            fill_value=0.0)
        ref.createVariable(
            'data',
            'f4',
            dimensions=('time', 'yc', 'xc'),
            zlib=True,
            fill_value=0.0)
        return ref

    def setUp(self):
        self.refs = [self.createRefFile('unittest%s.nc' % (str(i).zfill(2)))
                     for i in range(5)]
        [ref.sync() for ref in self.refs]
        self.ro_ref = self.createRefFile('ro_unittest.nc')
        self.ro_ref.sync()

    def tearDown(self):
        [ref.close() for ref in self.refs]

    def test_open_existent_file(self):
        # check if open an existent file.
        root, is_new = nc.open('unittest00.nc')
        self.assertFalse(is_new)
        self.assertEquals(len(root.roots), 1)
        self.assertFalse(root.read_only)

    def test_open_new_file(self):
        # delete the filename from the system
        filename = 'unittest-1.nc'
        if os.path.isfile(filename):
            os.remove(filename)
        # check if create and open a new file.
        root, is_new = nc.open(filename)
        self.assertTrue(is_new)
        self.assertEquals(len(root.roots), 1)
        self.assertFalse(root.read_only)

    def test_open_readonly_file(self):
        # delete the filename from the system
        filename = 'readonly.nc'
        if os.path.isfile(filename):
            os.chmod(filename, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        # check if create and open a new file.
        root, is_new = nc.open(filename)
        self.assertFalse(is_new)
        self.assertEquals(len(root.roots), 1)
        self.assertTrue(root.read_only)

    def test_open_multiple_files(self):
        # check if open the pattern selection using using a package instance.
        root, is_new = nc.open('unittest0*.nc')
        self.assertFalse(is_new)
        self.assertEquals(len(root.roots), 5)
        self.assertFalse(root.read_only)

    def test(self):
        pass


if __name__ == '__main__':
        unittest.main()
