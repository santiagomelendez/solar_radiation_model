Solar Radiation Model
=====================
[![Build Status](https://travis-ci.org/ecolell/solar_radiation_model.png?branch=master)](https://travis-ci.org/ecolell/solar_radiation_model) [![Coverage Status](https://coveralls.io/repos/ecolell/solar_radiation_model/badge.png)](https://coveralls.io/r/ecolell/solar_radiation_model)

Solar Radiation Model is a python service that provides you with a pipe and filter architecture. Further, it manage a field sensor database and an automatic downloader for satellite images.

Requirements
------------

If you want to deploy this repository with some default settings, you should have installed **python** (tested with versions 2.6 and 2.7) and **gmake**. If you Operative System doesn't have python, you can install it from sourcecode executing the next command:

	$ make install-python27

On any GNU/Linux or OSX you only need to execute the next bash command to setting up all the required requirements using the GNUMakefile.

	$ make sqlite3 deploy

At last you should configure a superuser frontend access. To do so, you should execute the next command and then fill (twice) the password field.

	$ make defaultsuperuser

Running
-------

There are 2 services, the **frontend** and the **backend**. First we recommend you to bootup the **frontend** using the command:

	$ make run

Now you can go to a browser on the same machine and use the address <http://localhost:8000/admin> to login to the service. You should complete the username field with "dev" and for the password field you should use your previously selected password.

Once you have set all the preferences, it's time to start the **backend** (the processing engine), you should use the next line:

	# python2.7 manage.py runbackend

About
-----

This software is developed by GERSolar. You can contact us to <gersolar.dev@gmail.com>.
