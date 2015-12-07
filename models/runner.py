from __future__ import print_function
from goesdownloader import instrument as goes
import os
from datetime import timedelta
import importlib
import glob
import pytz
from helpers import to_datetime
from collections import defaultdict
from core import pmap
import logging


class JobDescription(object):

    def __init__(self,
                 algorithm='heliosat',
                 data='data/*.nc',
                 static_file='static.nc',
                 product=None,
                 tile_cut={},
                 hard='cpu'):
        self.config = {
            'algorithm': 'models.{:s}'.format(algorithm),
            'data': data,
            'static_file': static_file,
            'product': product,
            'tile_cut': tile_cut,
            'hard': hard
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
        files = (glob.glob(filename)
                 if isinstance(filename, basestring) else filename)
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
        daylight = (lambda dt: localize(dt).hour >= 6
                    and localize(dt).hour <= 20)
        files = filter(lambda f: daylight(to_datetime(f)), files)
        return files

    def check_data(self):
        self.config['data'] = self.filter_data(self.config['data'])

    def run(self):
        estimated = 0
        if self.config['data']:
            m_lamb = lambda dt: '{:d}/{:d}'.format(dt.month, dt.year)
            months = list(set(pmap(m_lamb,
                                   pmap(to_datetime, self.config['data']))))
            pmap(lambda (k, o): logging.debug('{:s}: {:s}'.format(k, str(o))),
                 self.config.items())
            logging.info("Months: {:s}".format(str(months)))
            logging.info("Dataset: {:d} files.".format(
                len(self.config['data'])))
            algorithm = importlib.import_module(self.config['algorithm'])
            estimated, output = algorithm.run(**self.config)
            logging.info("Process finished.")
        return estimated, output


logging.basicConfig(level=logging.INFO)


def run(**config):
    diff = lambda dt, h: (dt - timedelta(hours=h))
    decimal = (lambda dt, h: diff(dt, h).hour + diff(dt, h).minute / 60. +
               diff(dt, h).second / 3600.)
    should_download = lambda dt: decimal(dt, 4) >= 5 and decimal(dt, 4) <= 20
    filenames = goes.download('noaa.gvarim', 'noaaadmin', 'data_argentina',
                              name='Argentina',
                              datetime_filter=should_download)
    print(filenames)
    # if filenames:
    #     work = JobDescription(data='data/goes13.*.BAND_01.nc')
    #     heliosat.workwith('data/goes13.*.BAND_01.nc')
