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
        files = glob(path + '/*.nc')
        dt = to_datetime(self.filename)
        days_ago = (dt - timedelta(days=30))
        files = filter(lambda f: days_ago < to_datetime(f) < dt, files)
        return files

    @property
    def time(self):
        self.times = map(lambda s: to_datetime(s), self.filenames_from_a_month_ago) 
        return self.times

    @property
    def decimalhour(self):
        int_to_decimalhour = (lambda time: time.hour +
                              time.minute/60.0 +
                              time.second/3600.0)
        result = map(int_to_decimalhour, self.time)
        return np.array(result)

    def calculate_slots(self):
        return np.round(self.decimalhour * self.images_per_hour).astype(int)


