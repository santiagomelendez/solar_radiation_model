import heliosat
from goesdownloader import instrument as goes
import os
import requests


def run():
    print os.environ
    filenames = goes.download('noaa.gvarim', 'noaaadmin', 'data_argentina',
                              name='Argentina')
    print filenames
    # if filenames:
    #     heliosat.workwith('data/goes13.*.BAND_01.nc')
