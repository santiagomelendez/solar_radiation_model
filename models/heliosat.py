#!/usr/bin/env python
import core
from core import pmap
import numpy as np
import glob
import os
from netcdf import netcdf as nc
from cache import Cache, Loader
from helpers import short
import logging


class Heliosat2(object):

    def __init__(self, config, strategy_type):
        self.config = config
        self.filenames = config['data']
        self.SAT_LON = -75.113
        # -75.3305 # longitude of sub-satellite point in degrees
        self.IMAGE_PER_HOUR = 2
        self.GOES_OBSERVED_ALBEDO_CALIBRATION = 1.89544 * (10 ** (-3))
        self.i0met = np.pi / self.GOES_OBSERVED_ALBEDO_CALIBRATION
        self.strategy_type = strategy_type
        self.cache = TemporalCache(self)

    def __del__(self):
        self.cache = None

    def create_1px_dimensions(self, root):
        nc.getdim(root, 'xc_k', 1)
        nc.getdim(root, 'yc_k', 1)
        nc.getdim(root, 'time', 1)

    def create_slots(self, loader, cache, strategy):
        self.create_1px_dimensions(cache)
        time = loader.time
        shape = list(time.shape)
        shape.append(1)
        strategy.times = time.reshape(tuple(shape))
        strategy.slots = cache.getvar('slots', 'i1', ('time', 'yc_k', 'xc_k'))
        strategy.slots[:] = strategy.calculate_slots(self.IMAGE_PER_HOUR)
        nc.sync(cache)

    def create_variables(self, loader, cache, strategy):
        self.create_slots(loader, cache, strategy)
        self.create_temporal(loader, cache, strategy)

    def create_temporal(self, loader, cache, strategy):
        create_f = lambda name, source: cache.getvar(name, 'f4', source=source)
        create = lambda name, source: cache.getvar(name, source=source)
        strategy.declination = create_f('declination', strategy.slots)
        strategy.solarangle = create_f('solarangle', loader.ref_data)
        nc.sync(cache)
        strategy.solarelevation = create('solarelevation', strategy.solarangle)
        strategy.excentricity = create_f('excentricity', strategy.slots)
        strategy.gc = create('gc', strategy.solarangle)
        strategy.atmosphericalbedo = create('atmosphericalbedo',
                                            strategy.solarangle)
        strategy.t_sat = create('t_sat', loader.ref_lon)
        strategy.t_earth = create('t_earth', strategy.solarangle)
        strategy.cloudalbedo = create('cloudalbedo', strategy.solarangle)
        nc.sync(cache)

    def update_temporalcache(self, loader, cache):
        logging.info("Updating temporal cache... ")
        self.strategy = self.strategy_type(self, loader, cache)
        self.strategy.update_temporalcache(loader, cache)

    def estimate_globalradiation(self, loader, cache):
        # There is nothing to do, if there isn't new cache and strategy setted.
        if hasattr(self, 'strategy'):
            logging.info("Obtaining the global radiation... ")
            output = OutputCache(self)
            self.strategy.estimate_globalradiation(loader, cache, output)
            output.dump()
            output = None
        cache.dump()

    def run_with(self, loader):
        self.estimate_globalradiation(loader, self.cache)


class AlgorithmCache(Cache):

    def __init__(self, algorithm):
        super(AlgorithmCache, self).__init__()
        self.algorithm = algorithm
        self.tile_config = self.algorithm.config['tile_cut']
        self.filenames = self.algorithm.filenames
        self.initialize_path(self.filenames)

    def __del__(self):
        super(AlgorithmCache, self).__del__()
        self.root = None


class TemporalCache(AlgorithmCache):

    def __init__(self, algorithm):
        super(TemporalCache, self).__init__(algorithm)
        self.update_cache(self.filenames)
        self.cache = Loader(pmap(self.get_cached_file, self.filenames),
                            tile_cut=self.tile_config)
        self.root = self.cache.root

    def initialize_path(self, filenames):
        self.path = '/'.join(filenames[0].split('/')[0:-1])
        self.temporal_path = self.algorithm.config['temporal_cache']
        self.index = {self.get_cached_file(v): v for v in filenames}
        if not os.path.exists(self.temporal_path):
            os.makedirs(self.temporal_path)

    def get_cached_file(self, filename):
        return '%s/%s' % (self.temporal_path, short(filename, None, None))

    def update_cache(self, filenames):
        self.clean_cache(filenames)
        self.extend_cache(filenames)

    def get_processed_files(self):
        files = glob.glob('%s/*.nc' % self.temporal_path)
        if files:
            with nc.loader(files, dimensions=self.tile_config) as cache:
                gc = nc.getvar(cache, 'gc')[:]
                files = filter(lambda f: np.any(gc[files.index(f), :] != 0),
                               files)
        return files

    def extend_cache(self, filenames):
        cached_files = self.get_processed_files()
        not_cached = filter(lambda f: self.get_cached_file(f)
                            not in cached_files,
                            filenames)
        if not_cached:
            loader = Loader(not_cached, self.tile_config)
            new_files = pmap(self.get_cached_file, not_cached)
            with nc.loader(new_files, dimensions=self.tile_config) as cache:
                self.algorithm.update_temporalcache(loader, cache)
            loader.dump()

    def clean_cache(self, exceptions):
        cached_files = glob.glob('%s/*.nc' % self.temporal_path)
        old_cache = filter(lambda f: self.index[f] not in exceptions,
                           cached_files)
        pmap(os.remove, old_cache)

    def getvar(self, *args, **kwargs):
        name = args[0]
        if name not in self._attrs.keys():
            tmp = list(args)
            tmp.insert(0, self.cache.root)
            self._attrs[name] = nc.getvar(*tmp, **kwargs)
        return self._attrs[name]


class OutputCache(AlgorithmCache):

    def __init__(self, algorithm):
        super(OutputCache, self).__init__(algorithm)
        self.output = Loader(pmap(self.get_output_file, self.filenames),
                             tile_cut=self.tile_config)
        self.root = self.output.root
        with nc.loader(self.filenames, dimensions=self.tile_config) as images:
            map(algorithm.create_1px_dimensions, self.root.roots)
            self.root.getvar('time', source=images.getvar('time'))
            self.root.getvar('cloudindex',
                             'f4', source=images.getvar('data'))
            self.root.getvar('globalradiation',
                             'f4', source=images.getvar('data'))

    def initialize_path(self, filenames):
        self.path = '/'.join(filenames[0].split('/')[0:-1])
        self.output_path = self.algorithm.config['product']
        self.index = {self.get_output_file(v): v for v in filenames}
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def get_output_file(self, filename):
        return '%s/%s' % (self.output_path, short(filename, None, None))


def run(**config):
    loader = Loader(config['data'], tile_cut=config['tile_cut'])
    core.config = config
    from core import geo
    algorithm = Heliosat2(config, geo.strategy)
    algorithm.run_with(loader)
    algorithm = None
    loader.dump()
