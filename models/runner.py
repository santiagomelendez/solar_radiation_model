from __future__ import print_function
from goesdownloader import instrument as goes
from cpu import CPUStrategy
from datetime import datetime, timedelta
from helpers import to_datetime
from temporal_serie import TemporalSerie
from cache import StaticCache, Cache, OutputCache
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
        self.load_data()

    def load_data(self):
        static = self.config['static_file']
        if isinstance(self.config['data'], (list, str)):
            self.config['data'] = Cache(self.config['data'],
                                        tile_cut=self.config['tile_cut'],
                                        read_only=True)
        self.config['filenames'] = self.config['data'].filenames
        if isinstance(static, str):
            self.config['static_file'] = StaticCache(static,
                                                     self.config['filenames'],
                                                     self.config['tile_cut'])
        self.config['product'] = OutputCache(self.config['product'],
                                             self.config['tile_cut'],
                                             self.config['filenames'])

    @classmethod
    def filter_data(cls, filename):
        data = TemporalSerie(filename).get()
        return data

    def check_data(self):
        if isinstance(self.config['data'], (str, list)):
            self.config['data'] = self.filter_data(self.config['data'])

    def run(self):
        if isinstance(self.config['data'], (str, list)):
            m_lamb = lambda dt: '{:d}/{:d}'.format(dt.month, dt.year)
            months = list(set(map(m_lamb,
                                  map(to_datetime,
                                      self.config['data']))))
            map(lambda (k, o): logging.debug(
                '{:s}: {:s}'.format(k, str(o))), self.config.items())
            logging.info("Months: {:s}".format(str(months)))
            logging.info("Dataset: {:d} files.".format(
                len(self.config['data'])))
        time = self.config['data'].time
        data = self.config['data'].data
        lat = self.config['static_file'].lat
        lon = self.config['static_file'].lon
        dem = self.config['static_file'].dem
        linke = self.config['static_file'].linke
        start = datetime.now()
        cpu = CPUStrategy(time)
        data = cpu.getcalibrateddata(self.config['data'])
        cloudindex, globalradiation = cpu.estimate_globalradiation(lat, lon,
                                                                   dem, linke,
                                                                   data)
        output = self.config['product']
        output.ref_cloudindex[:] = cloudindex
        output.ref_globalradiation[:] = globalradiation
        logging.info("Process finished.")
        end = datetime.now()
        return (end - start).total_seconds(), output


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


def run_heliosat(**config):
    job = JobDescription(data='data/goes13.2015.048.143733.BAND_01.nc',
                         product='products/estimated',
                         static_file='static.nc')
    job.run()
