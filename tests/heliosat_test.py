import unittest
from models.heliosat import main
import numpy as np
from netcdf import netcdf as nc
import datetime
import os
from goescalibration import instrument


class TestHeliosat(unittest.TestCase):

    def create_mook(self, filename):
        ma = np.matrix('5.1 3.2 1.3 -1.4 -3.5 -4.6;'
                       '5.2 3.3 1.4 -1.5 -3.6 -4.7;'
                       '5.4 3.5 1.5 -1.6 -3.7 -4.8;'
                       '5.6 3.7 1.8 -1.8 -3.9 -4.9 ')
        x_m, y_m = ma.shape
        root, _ = nc.open(filename)
        nc.getdim(root, 'auditCount', 2)
        nc.getdim(root, 'auditSize', 80)
        nc.getdim(root, 'xc', x_m)
        nc.getdim(root, 'yc', y_m)
        nc.getdim(root, 'time')
        var = nc.getvar(root,
            'time',
            'i4',
            dimensions=('time',),
            fill_value=0)
        var[0] = 1
        var = nc.getvar(root,
            'lat',
            'f4',
            dimensions=('yc', 'xc'),
            fill_value=0.0)
        var[:] = 1
        var = nc.getvar(root,
            'lon',
            'f4',
            dimensions=('yc', 'xc'),
            fill_value=0.0)
        var[:] = 1
        var = nc.getvar(root,
            'data',
            'f4',
            dimensions=('time', 'yc', 'xc'),
            fill_value=0.0)
        var[0, :] = 1
        var = nc.getvar(root,
            'auditTrail',
            'S1',
            dimensions=('auditCount', 'auditSize'),
            fill_value=0.0)
        audit = np.array([['1', '4', '0', '0', '1', ' ', '2', '3', '2', '3',
                           '5', '2', ' ', 'I', 'M', 'G', 'C', 'O', 'P', 'Y',
                           ' ', 'D', 'E', 'L', 'I', 'V', 'E', 'R', 'Y', '/',
                           'I', 'N', '1', '2', '6', '7', '8', '3', '3', '1',
                           '9', '4', '.', '1', ' ', 'D', 'E', 'L', 'I', 'V',
                           'E', 'R', 'Y', '/', 'N', 'C', '1', '2', '6', '7',
                           '8', '3', '3', '1', '9', '4', '.', '1', ' ', 'L',
                           'I', 'N', 'E', 'L', 'E', '=', '1', '1', '0', '1'],
                          [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
                           ' ', ' ', ' ', '0', ' ', '1', '6', '8', '5', '2',
                           ' ', 'I', ' ', 'P', 'L', 'A', 'C', 'E', '=', 'U',
                           'L', 'E', 'F', 'T', ' ', 'B', 'A', 'N', 'D', '=',
                           '1', ' ', 'D', 'O', 'C', '=', 'Y', 'E', 'S', ' ',
                           'M', 'A', 'G', '=', '-', '1', ' ', '-', '1', ' ',
                           'S', 'I', 'Z', 'E', '=', '1', '0', '6', '7', ' ',
                           '2', '1', '6', '6', ' ', ' ', ' ', ' ', ' ', ' ']])
        var[:] = audit
        nc.close(root)

    def filenames(self):
        amount = 5
        date = datetime.datetime(2013, 3, 1, 10, 0, 0)
        dates = map(lambda i: date + datetime.timedelta(hours=i*2),
                    range(amount))
        name = lambda d: 'goes13.%s.BAND_01.nc' % d.strftime('%Y.%j.%H%m%S')
        return map(name, dates)

    def setUp(self):
        self.files = self.filenames()
        # It create a small collection of small images.
        map(self.create_mook, self.files)
        list(map(instrument.calibrate, self.files))

    def tearDown(self):
        os.system('rm *.nc')

    def test_main(self):
        main.workwith(2013, 3, 'goes13.2013.*.BAND_01.nc')


if __name__ == '__main__':
    unittest.run()
