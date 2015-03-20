import heliosat
from goesdownloader import instrument as goes
import os
import requests
from datetime import datetime, timedelta


def run():
    diff = lambda dt, h: (dt - timedelta(hours=h))
    decimal = lambda dt, h: diff(dt, h).hour + diff(dt, h).minute / 60. + diff(dt, h).second / 3600.
    should_download = lambda dt: decimal(dt, 4) >= 5 and decimal(dt, 4) <= 20
    filenames = goes.download('noaa.gvarim', 'noaaadmin', 'data_argentina',
                              name='Argentina', datetime_filter=should_download)
    print filenames
    # if filenames:
    #     heliosat.workwith('data/goes13.*.BAND_01.nc')
