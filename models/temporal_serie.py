from __future__ import print_function
from datetime import datetime, timedelta
from netcdf import netcdf as nc
from glob import glob
import numpy as np
from helpers import to_datetime


class TemporalSerie(object):

    def __init__(self, filename, images_per_hour=2):
        self.filename = filename if isinstance(filename, str) else filename[0]
        self.images_per_hour = images_per_hour

    def get(self):
        data = self.get_conditional_data()
        data.append(self.filename)
        return data

    def get_conditional_data(self):
        slots = self.calculate_slots()
        condition = ((slots >= 20) & (slots <= 28))
        serie = []
        map(lambda i: serie.append(self.filenames_from_a_month_ago[i]),
            np.where(condition)[0])
        return serie

    @property
    def filenames_from_a_month_ago(self):
        path = '/'.join(self.filename.split('/')[:-1])
        dt = to_datetime(self.filename)
        a_month_ago = []
        jdb = lambda jd: (dt - timedelta(days=jd)).strftime('%Y.%j')
        days_before = lambda day: a_month_ago.extend(
            glob(path + '/*.%s.*.nc' % jdb(day)))
        days_before(1)
        return a_month_ago

    @property
    def time(self):
        with nc.loader(self.filenames_from_a_month_ago) as loader:
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

    def calculate_slots(self):
        return np.round(self.decimalhour * self.images_per_hour).astype(int)


