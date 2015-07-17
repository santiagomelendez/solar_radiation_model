import heliosat
from goesdownloader import instrument as goes
import os
import requests
from datetime import datetime, timedelta
import importlib
import glob
import pytz
from helpers import to_datetime, show
from collections import defaultdict
from core import pmap


class JobDescription(object):

    def __init__(self,
                 algorithm = 'heliosat',
                 data='data/*.nc',
                 static_file = 'static.nc',
                 temporal_cache = 'temporal_cache',
                 product = 'product/estimated',
                 tile_cut = {}):
        self.config = {
            'algorithm': 'models.%s' % algorithm,
            'data': data,
            'static_file': static_file,
            'temporal_cache': temporal_cache,
            'product': product,
            'tile_cut': tile_cut
        }
        self.check_data()

    def filter_wrong_sized_data(self, files):
        size = lambda f: os.stat(f).st_size
        sizes = defaultdict(int)
        for f in files:
            sizes[size(f)] += 1
        more_freq = max(sizes.values())
        right_size = filter(lambda (s, freq): freq == more_freq,
                            sizes.items())[0][0]
        return filter(lambda f: size(f) == right_size, files)

    def filter_data(self, filename):
        files = glob.glob(filename) if isinstance(filename, str) else filename
        if not files:
            return []
        last_dt = to_datetime(max(files))
        a_month_ago = (last_dt - timedelta(days=30)).date()
        gmt = pytz.timezone('GMT')
        local = pytz.timezone('America/Argentina/Buenos_Aires')
        localize = lambda dt: (gmt.localize(dt)).astimezone(local)
        in_the_last_month = lambda f: to_datetime(f).date() >= a_month_ago
        files = filter(in_the_last_month, files)
        files = self.filter_wrong_sized_data(files)
        daylight = lambda dt: localize(dt).hour >= 6 and localize(dt).hour <= 20
        files = filter(lambda f: daylight(to_datetime(f)), files)
        return files

    def check_data(self):
        self.config['data'] = self.filter_data(self.config['data'])

    def run(self):
        if self.config['data']:
            months = list(set(pmap(lambda dt: '%i/%i' % (dt.month, dt.year),
                                   pmap(to_datetime, self.config['data']))))
            show("=======================")
            show("Months: ", months)
            show("Dataset: ", len(self.config['data']), " files.")
            show("-----------------------\n")
            algorithm = importlib.import_module(self.config['algorithm'])
            algorithm.run(**self.config)
            show("Process finished.\n")


def run():
    diff = lambda dt, h: (dt - timedelta(hours=h))
    decimal = lambda dt, h: diff(dt, h).hour + diff(dt, h).minute / 60. + diff(dt, h).second / 3600.
    should_download = lambda dt: decimal(dt, 4) >= 5 and decimal(dt, 4) <= 20
    filenames = goes.download('noaa.gvarim', 'noaaadmin', 'data_argentina',
                              name='Argentina', datetime_filter=should_download)
    print filenames
    # if filenames:
    #     work = JobDescription(data='data/goes13.*.BAND_01.nc')
    #     heliosat.workwith('data/goes13.*.BAND_01.nc')
