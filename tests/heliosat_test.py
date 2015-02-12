import unittest
from models.heliosat import main
import numpy as np
from netcdf import netcdf as nc
import datetime
import os
import glob


class TestHeliosat(unittest.TestCase):

    def setUp(self):
        os.system('cp -rf data mock_data')
        self.files = glob.glob('mock_data/goes13.*.BAND_01.nc')

    def tearDown(self):
        os.system('rm -rf mock_data')

    def test_main(self):
        main.workwith(2013, 3, 'mock_data/goes13.2015.*.BAND_01.nc')


if __name__ == '__main__':
    unittest.run()
