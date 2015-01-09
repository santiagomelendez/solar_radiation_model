Solar Radiation Model
=====================

[![Gitter](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/gersolar/solar_radiation_model?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

[![GNU/AGPL License](http://www.gnu.org/graphics/agplv3-88x31.png)](https://github.com/gersolar/solar_radiation_model/blob/master/GNU-AGPL-3.0.txt) [![Build Status](https://travis-ci.org/gersolar/solar_radiation_model.png?branch=master)](https://travis-ci.org/gersolar/solar_radiation_model) [![Coverage Status](https://coveralls.io/repos/gersolar/solar_radiation_model/badge.png?branch=master)](https://coveralls.io/r/gersolar/solar_radiation_model?branch=master) [![Code Health](https://landscape.io/github/gersolar/solar_radiation_model/master/landscape.png)](https://landscape.io/github/gersolar/solar_radiation_model/master)
[![Supported Python versions](https://pypip.in/py_versions/solar_radiation_model/badge.svg)](https://pypi.python.org/pypi/solar_radiation_model/) [![Stories in Ready](https://badge.waffle.io/gersolar/solar_radiation_model.png?label=ready&title=Ready)](https://waffle.io/gersolar/solar_radiation_model)

Solar Radiation Model is a python script that estimates the solar radiation at the soil level.

Requirements
------------

If you want to deploy this repository with the default settings, on any GNU/Linux or OSX system you just need to execute the next bash command to setting up all the requirements (GNUMakefile should have been installed to this point).

	$ PYVERSION=2.7 make sqlite3 deploy

But, if you want use **postgresql** instead of sqlite3, you should execute the next bash command:

	$ PYVERSION=2.7 make postgres deploy

All the testing are made in **python2.7**. If you haven't installed **python2.7** it is automatically installed with the previous command.

Last, you should configure a superuser access to the **frontend**. To do so, you should execute the next command and then fill the password field.

	$ make defaultsuperuser

Running
-------

There are 2 services, the **frontend** and the **backend**. First we recommend you to bootup the **frontend** using the command:

	$ make run

Now you can go to a browser on the same machine and use the address <http://localhost:8000/admin> to login to the service. You should complete the username field with "dev" and in the password field you should use your previously selected password.

Once you have set all the preferences, it's time to start the **backend** (the processing engine), you should use the next line:

	$ make runbackend

About
-----

This software is developed by [GERSolar](http://www.gersol.unlu.edu.ar/). You can contact us to <gersolar.dev@gmail.com>.
