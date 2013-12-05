Solar Radiation Model
=====================
[![Build Status](https://travis-ci.org/ecolell/solar_radiation_model.png?branch=master)](https://travis-ci.org/ecolell/solar_radiation_model)

Solar Radiation Model is a python service that provides you with pipe & filter architecture. Further, it manage a field sensor database and an automatic downloader for satellite images.

Requirements
------------

If you want to deploy this repository with some default settings, you should have installed **python2.7** and **gmake**.

On any linux you only need to execute the next bash command to setting up all the required requirements using the GNUMakefile.

	$ sudo make sqlite3 deploy

Running
-------

There are 2 things to run: the **frontend** and the **backend**. First we recommend to you to run the **frontend** using the command:

	# sudo make run

Now you can go to a browser on the same machine and use the address "http://localhost/admin" to login to the service. Remember to use the previous data.

Play a lot, and try to understand each part of the page.

After that, when you are ready to start the *backend* (the download engine), you should use the next line:

	# python2.7 manage.py runbackend

ABOUT
-----

One branch of this software is still being developed by GERSolar. If you have any questions or ideas email us to <gersolar.dev@gmail.com>.