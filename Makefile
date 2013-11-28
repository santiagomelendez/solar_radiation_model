deploy: pip-requirements
	cd imagedownloader && python manage.py syncdb --noinput
	cd imagedownloader && python manage.py migrate

bin-sqlite3:
	wget http://www.sqlite.org/2013/sqlite-autoconf-3080100.tar.gz > /dev/null
	tar xvfz sqlite-autoconf-3080100.tar.gz > /dev/null
	cd sqlite-autoconf-3080100 && ./configure > /dev/null
	cd sqlite-autoconf-3080100 && make > /dev/null
	cd sqlite-autoconf-3080100 && sudo make install > /dev/null

lib-hdf5:
	wget http://www.hdfgroup.org/ftp/HDF5/current/src/hdf5-1.8.12.tar.gz > /dev/null
	tar xzvvf hdf5-1.8.12.tar.gz > /dev/null
	cd hdf5-1.8.12 && ./configure --prefix=/usr/local --enable-shared --enable-hl > /dev/null
	cd hdf5-1.8.12 && make -j 2 > /dev/null
	cd hdf5-1.8.12 && sudo make install > /dev/null

lib-netcdf4:
	wget ftp://ftp.unidata.ucar.edu/pub/netcdf/netcdf-4.3.1-rc4.tar.gz > /dev/null
	tar xzvvf netcdf-4.3.1-rc4.tar.gz > /dev/null
	cd netcdf-4.3.1-rc4 && LDFLAGS=-L/usr/local/lib CPPFLAGS=-I/usr/local/include sh ./configure --enable-netcdf-4 --enable-dap --enable-shared --prefix=/usr/local > /dev/null
	cd netcdf-4.3.1-rc4 && make -j 2 > /dev/null
	cd netcdf-4.3.1-rc4 && sudo make install > /dev/null

src-aspects:
	wget http://www.cs.tut.fi/~ask/aspects/python-aspects-1.3.tar.gz > /dev/null
	tar xzvvf python-aspects-1.3.tar.gz > /dev/null
	cd python-aspects-1.3 && sudo make && sudo make install > /dev/null
	cp python-aspects-1.3/aspects.py imagedownloader/aspects.py

src-postgres:
	--without-readline --without-zlib

pip-requirements: bin-sqlite3 lib-hdf5 lib-netcdf4 src-aspects
	sudo ldconfig
	pip install -r imagedownloader/requirements.txt > /dev/null
