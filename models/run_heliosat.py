from __future__ import print_function
from datetime import datetime, timedelta
from netcdf import netcdf as nc
import importlib
import glob
import numpy as np
from helpers import to_datetime
from cache import StaticCache, Cache, OutputCache
import logging


class JobDescription(object):

    def __init__(self,
                 algorithm='heliosat',
                 data='data/*.nc',
                 static_file='static.nc',
                 product='products/estimated',
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
        self.image = self.config['data']
        self.prepare_data()
        self.load_data()

    def load_data(self):
        static = self.config['static_file']
        if isinstance(self.config['data'], (list, str)):
            start = datetime.now()
            self.config['data'] = Cache(self.config['data'],
                                        tile_cut=self.config['tile_cut'],
                                        read_only=True)
        self.config['filenames'] = self.config['data'].filenames
        end = datetime.now()
        finish = (end - start).total_seconds()
        print('total time of Cache: ', finish)
        if isinstance(static, str):
            start = datetime.now()
            self.config['static_file'] = StaticCache(static,
                                                     self.config['filenames'],
                                                     self.config['tile_cut'])
            end = datetime.now()
            finish = (end - start).total_seconds()
            print('total time of StaticCache: ', finish)
        start = datetime.now()
        self.config['product'] = OutputCache(self.config['product'],
                                             self.config['tile_cut'],
                                             self.config['filenames'][-1:])
        end = datetime.now()
        finish = (end - start).total_seconds()
        print('total time of OutputCache: ', finish)

    @property
    def time(self):
        with nc.loader(self.config['data']) as loader:
            self.times = nc.getvar(loader, 'time')[:]
        return self.times

    @property
    def decimalhour(self):
        int_to_dt = lambda t: datetime.utcfromtimestamp(t)
        int_to_decimalhour = (lambda time: int_to_dt(time).hour +
                              int_to_dt(time).minute/60.0 +
                              int_to_dt(time).second/3600.0)
        result = map(int_to_decimalhour, self.time)
        return np.array(result)

    def calculate_slots(self, images_per_hour):
        return np.round(self.decimalhour * images_per_hour).astype(int)

    def get_a_30_days_ago_data(self):
        filename = self.config['data']
        path = '/'.join(filename.split('/')[:-1])
        dt = to_datetime(filename)
        a_month_ago = []
        jdb = lambda jd: (dt - timedelta(days=jd)).strftime('%Y.%j')
        days_before = lambda day: a_month_ago.extend(
            glob.glob(path + '/*.%s.*.nc' % jdb(day)))
        map(days_before, range(1, 31))
        files = sorted(a_month_ago)
        self.config['data'] = files
        slots = self.calculate_slots(2)
        condition = ((slots >= 20) & (slots <= 28))
        temporal_serie = []
        map(lambda i: temporal_serie.append(files[i]), np.where(condition)[0])
        return sorted(temporal_serie)

    def prepare_data(self):
        self.config['data'] = self.get_a_30_days_ago_data()
        self.config['data'].append(self.image)

    def run(self):
        estimated = 0
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
        algorithm = importlib.import_module(self.config['algorithm'])
        estimated, output = algorithm.run(**self.config)
        logging.info("Process finished.")
        return estimated, output


logging.basicConfig(level=logging.INFO)


def run():
    work = JobDescription(data='data/goes13.2015.048.143733.BAND_01.nc')
    intern_estimated, output = work.run()
    estimated = intern_estimated / 3600.
    print("estimate_globalradiation time: {:.2f} hours.".format(estimated))
