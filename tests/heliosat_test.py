import unittest
from models.heliosat import main
import numpy as np
from netcdf import netcdf as nc
import datetime
import os
import glob


class TestHeliosat(unittest.TestCase):

    def setUp(self):
        self.files = glob.glob('data/goes13.*.BAND_01.nc')

    def test_main(self):
        pass
        #main.workwith(2013, 3, 'data/goes13.2015.*.BAND_01.nc')


if __name__ == '__main__':
    unittest.run()
