deploy: pip-aspects lib-hdf5 lib-netcdf4
	sudo ldconfig
	#apt-get install libhdf5-serial-dev python-netcdf

lib-hdf5:
	wget http://www.hdfgroup.org/ftp/HDF5/current/src/hdf5-1.8.12.tar.gz
	tar xzvvf hdf5-1.8.12.tar.gz
	cd hdf5-1.8.12 && ./configure --prefix=/usr/local --enable-shared --enable-hl
	cd hdf5-1.8.12 && make -j 2
	cd hdf5-1.8.12 && sudo make install

lib-netcdf4:
	wget ftp://ftp.unidata.ucar.edu/pub/netcdf/netcdf-4.3.1-rc4.tar.gz
	tar xzvvf netcdf-4.3.1-rc4.tar.gz
	cd netcdf-4.3.1-rc4 && LDFLAGS=-L/usr/local/lib CPPFLAGS=-I/usr/local/include sh ./configure --enable-netcdf-4 --enable-dap --enable-shared --prefix=/usr/local
	cd netcdf-4.3.1-rc4 && make -j 2
	cd netcdf-4.3.1-rc4 && sudo make install

pip-aspects:
	wget https://bitbucket.org/pypa/setuptools/raw/bootstrap/ez_setup.py
	sudo python ez_setup.py
	easy_install https://pypi.python.org/packages/2.5/p/python-aspects/python_aspects-1.1-py2.5.egg#md5=a99122b2294d12bac84f67e6029331b9
