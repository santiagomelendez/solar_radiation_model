#!/usr/bin/env python
import core
from core import pmap
import numpy as np
import os
from netcdf import netcdf as nc
from cache import StaticCache, Cache
from helpers import short
import logging
import importlib
from datetime import datetime


class Heliosat2(object):

    def __init__(self, config, strategy_type, loader):
        self.config = config
        self.filenames = config['data']
        self.SAT_LON = -75.113
        # -75.3305 # longitude of sub-satellite point in degrees
        self.IMAGE_PER_HOUR = 2
        self.GOES_OBSERVED_ALBEDO_CALIBRATION = 1.89544 * (10 ** (-3))
        self.i0met = np.pi / self.GOES_OBSERVED_ALBEDO_CALIBRATION
        self.strategy = strategy_type(self, loader)
        self.loader = loader
        self.static = StaticCache(self)

    def estimate_globalradiation(self):
        logging.info("Obtaining the global radiation... ")
        output = OutputCache(self)
        self.strategy.estimate_globalradiation(self.static,
                                               self.loader, output)
        return output

    def run_with(self):
        logging.info("Take begin time.")
        begin = datetime.now()
        output = self.estimate_globalradiation()
        end = datetime.now()
        logging.info("Take end time.")
        return (end - begin).total_seconds(), output


class OutputCache(Cache):

    def __init__(self, algorithm):
        super(OutputCache, self).__init__(algorithm.filenames,
                                          algorithm.config['tile_cut'])
        self.algorithm = algorithm
        self.initialize_variables(self.filenames)

    def create_1px_dimensions(self, root):
        nc.getdim(root, 'xc_k', 1)
        nc.getdim(root, 'yc_k', 1)
        nc.getdim(root, 'time', 1)

    def initialize_path(self, filenames, images):
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        self.output = Cache(pmap(self.get_output_file, self.filenames),
                            tile_cut=self.tile_cut)
        self.root = self.output.root
        map(self.create_1px_dimensions, self.root.roots)
        self.root.getvar('time', source=images.getvar('time'))
        self.root.getvar('cloudindex', 'f4', source=images.getvar('data'))
        self.root.getvar('globalradiation', 'f4', source=images.getvar('data'))

    def initialize_variables(self, filenames):
        self.path = '/'.join(filenames[0].split('/')[0:-1])
        self.output_path = self.algorithm.config['product']
        with nc.loader(self.filenames, dimensions=self.tile_cut) as images:
            if self.output_path:
                self.initialize_path(filenames, images)
            else:
                data_shape = images.getvar('data').shape
                self.time = np.zeros(images.getvar('time').shape)
                self.ref_cloudindex = np.zeros(data_shape)
                self.cloudindex = self.ref_cloudindex
                self.ref_globalradiation = np.zeros(data_shape)
                self.globalradiation = self.ref_globalradiation

    def get_output_file(self, filename):
        return '%s/%s' % (self.output_path, short(filename, None, None))


def run(**config):
    loader = Cache(config['data'], tile_cut=config['tile_cut'],
                   read_only=True)
    config = core.check_hard(config)
    geo = importlib.import_module('models.%s' % config['hard'])
    algorithm = Heliosat2(config, geo.strategy, loader)
    return algorithm.run_with()
