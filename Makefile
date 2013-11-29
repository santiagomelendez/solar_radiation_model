deploy: pip-requirements
	cd imagedownloader && python manage.py syncdb --noinput
	cd imagedownloader && python manage.py migrate

bin-sqlite3:
	wget http://www.sqlite.org/2013/sqlite-autoconf-3080100.tar.gz
	tar xvfz sqlite-autoconf-3080100.tar.gz 
	cd sqlite-autoconf-3080100 && ./configure
	cd sqlite-autoconf-3080100 && make
	cd sqlite-autoconf-3080100 && sudo make install

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

src-aspects:
	wget http://www.cs.tut.fi/~ask/aspects/python-aspects-1.3.tar.gz
	tar xzvvf python-aspects-1.3.tar.gz 
	cd python-aspects-1.3 && sudo make && sudo make install
	cp python-aspects-1.3/aspects.py imagedownloader/aspects.py

src-postgres:
	--without-readline --without-zlib

pip-requirements: bin-sqlite3 lib-hdf5 lib-netcdf4 src-aspects
	sudo ldconfig
	#pip install -r imagedownloader/requirements.txt
