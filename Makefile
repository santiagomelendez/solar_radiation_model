deploy: lib-hdf5 lib-netcdf4
	sudo ldconfig
	#apt-get install libhdf5-serial-dev python-netcdf

lib-hdf5:
	wget http://www.hdfgroup.org/ftp/HDF5/current/src/hdf5-1.8.12.tar.gz
	tar xzvvf hdf5-1.8.12.tar.gz
	cd hdf5-1.8.12
	sh ./configure --prefix=/usr/local --enable-shared --enable-hl
	make -j 2
	sudo make install
	cd ..

lib-netcdf4:
	wget ftp://ftp.unidata.ucar.edu/pub/netcdf/netcdf-4.3.1-rc4.tar.gz
	tar xzvvf netcdf-4.3.1-rc4.tar.gz
	cd netcdf-4.3.1-rc4
	LDFLAGS=-L/usr/local/lib CPPFLAGS=-I/usr/local/include sh ./configure --enable-netcdf-4 --enable-dap --enable-shared --prefix=/usr/local
	make -j 2
	sudo make install
	cd ..
