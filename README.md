  Solar Radiation Model
=====================

[![Gitter](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/gersolar/solar_radiation_model?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

[![License](https://img.shields.io/pypi/l/solar_radiation_model.svg)](https://raw.githubusercontent.com/gersolar/solar_radiation_model/master/LICENSE) [![Downloads](https://img.shields.io/pypi/dm/solar_radiation_model.svg)](https://pypi.python.org/pypi/solar_radiation_model/) [![Build Status](https://travis-ci.org/gersolar/solar_radiation_model.svg?branch=master)](https://travis-ci.org/gersolar/solar_radiation_model) [![Coverage Status](https://coveralls.io/repos/gersolar/solar_radiation_model/badge.png)](https://coveralls.io/r/gersolar/solar_radiation_model) [![PyPI version](https://badge.fury.io/py/solar_radiation_model.svg)](http://badge.fury.io/py/solar_radiation_model)
[![Stories in Ready](https://badge.waffle.io/gersolar/solar_radiation_model.png?label=ready&title=Ready)](https://waffle.io/gersolar/solar_radiation_model)

Solar Radiation Model is a python script that estimates the solar radiation at the soil level.


Requirements
------------

If you want to use this library on any GNU/Linux or OSX system you just need to execute:

    $ pip install solar_radiation_model

If you want to improve this library, you should download the [github repository](https://github.com/gersolar/solar_radiation_model) and execute:

    $ make deploy

On Ubuntu Desktop there are some other libraries not installed by default (zlibc curl libssl0.9.8 libbz2-dev libxslt*-dev libxml*-dev) which may need to be installed to use these library. Use the next command to automate the installation of the additional C libraries:

	$ make ubuntu deploy

or:

    $ make osx deploy


Testing
-------

To test all the project you should use the command:

    $ make test

If you want to help us or report an issue join to us through the [GitHub issue tracker](https://github.com/gersolar/solar_radiation_model/issues).


Example
-------

To use the library, create a JobDescription instance using a config dictionary:

```python
    from models import JobDescription
    config = {
        'algorithm': 'heliosat',
        'static_file': 'static.nc',
        'data': 'data/goes13.2015.*.nc',
        'tile_cut': {
            'xc': [10, 15],
            'yc': [20, 45]},
        'hard': 'gpu',
    }
    job = JobDescription(**config)
    elapsed_time, output = job.run()
    print output.time, output.cloudindex, output.globalradiation
```


About
-----

This software is developed by [GERSolar](http://www.gersol.unlu.edu.ar/). You can contact us to [gersolar.dev@gmail.com](mailto:gersolar.dev@gmail.com).
