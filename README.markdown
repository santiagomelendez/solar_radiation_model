Solar Radiation Model
=====================
[![Build Status](https://travis-ci.org/ecolell/solar_radiation_model.png?branch=master)](https://travis-ci.org/ecolell/solar_radiation_model)

Solar Radiation Model is a python service that provides you with a pipe and filter architecture. Further, it manage a field sensor database and an automatic downloader for satellite images.

Requirements
------------

If you want to deploy this repository with some default settings, you should have installed **python2.7** and **gmake**.

On any linux you only need to execute the next bash command to setting up all the required requirements using the GNUMakefile.

	$ make sqlite3 deploy

But, if you have got installed pip, you should execute the next command:

	$ make sqlite3 deploy-withoutpip

At last you should configure a superuser access to the fron end. To do so, you should execute the next command and then fill (twice) the password field.

	$ make defaultsuperuser

Running
-------

There are 2 things to run: the **frontend** and the **backend**. First we recommend you to run the **frontend** using the command:

	# make run

Now you can go to a browser on the same machine and use the address <http://localhost:8000/admin> to login to the service. You should complete the username field with "dev" and for the password field you should use your previously selected password.

Once you have set all the preferences, it's time to run the **backend** (the processing engine), you should use the next line:

	# python2.7 manage.py runbackend

About
-----

This software is developed by GERSolar. You can contact us to <gersolar.dev@gmail.com>.
