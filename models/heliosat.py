#!/usr/bin/env python
import core
import numpy as np
import logging
import importlib
from datetime import datetime


class Heliosat2(object):

    def __init__(self, config, strategy_type):
        self.config = config
        self.filenames = config['filenames']
        self.init_constants()
        self.loader = config['data']
        self.strategy = strategy_type(self, self.loader)
        self.static = config['static_file']
        self.output = config['product']

    def init_constants(self):
        self.SAT_LON = -75.113
        # -75.3305 # longitude of sub-satellite point in degrees
        self.IMAGE_PER_HOUR = 2
        self.GOES_OBSERVED_ALBEDO_CALIBRATION = 1.89544 * (10 ** (-3))
        self.i0met = np.pi / self.GOES_OBSERVED_ALBEDO_CALIBRATION

    def estimate_globalradiation(self):
        logging.info("Obtaining the global radiation... ")
        self.strategy.estimate_globalradiation(self.static,
                                               self.loader, self.output)
        return self.output

    def run_with(self):
        logging.info("Take begin time.")
        begin = datetime.now()
        output = self.estimate_globalradiation()
        end = datetime.now()
        logging.info("Take end time.")
        return (end - begin).total_seconds(), output


def run(**config):
    config = core.check_hard(config)
    geo = importlib.import_module('models.{:s}'.format(config['hard']))
    algorithm = Heliosat2(config, geo.strategy)
    return algorithm.run_with()
