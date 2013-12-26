OS:=$(shell uname -s)
download = [ ! -f $(1) ] && echo "[ downloading  ] $(1)" && curl -O $(2)/$(1) || echo "[ downloaded   ] $(1)"
unpack = [ ! -d $(2) ] && echo "[ unpacking    ] $(1)" && tar xzf $(1) || echo "[ unpacked     ] $(1)"

define get
	@ $(call download,$(2),$(3))
	@ $(call unpack,$(2),$(1))
endef

define compile
	@ cd $(1) && \
	([ -f ./configure ] && echo "[ configuring  ] $(1)" && ($(2) sh ./configure $(3) 2>&1) >> ../tracking.log || echo "[ configured   ] $(1)") && \
	echo "[ compiling    ] $(1)" && \
	(make -j 2 2>&1) >> ../tracking.log && \
	echo "[ installing   ] $(1)" && \
	(sudo make install 2>&1) >> ../tracking.log
endef

define install
	@ $(call get,$(1),$(2),$(3))
	@ $(call compile,$(1))
endef

update_shared_libs=sudo ldconfig

ifeq ($(OS), Darwin)
	update_shared_libs=
	LIBSQLITE3=libsqlite3.0.dylib
	LIBHDF5=libhdf5_hl.8.dylib
	LIBNETCDF=libnetcdf.7.dylib
endif
ifeq ($(OS), Linux)
	#DISTRO=$(shell lsb_release -si)
	#ifeq ($(DISTRO), CentOS)
	#endif
	LIBSQLITE3=libsqlite3.so.0.8.6
	LIBHDF5=libhdf5.so.8.0.1
	LIBNETCDF=libnetcdf.so.7.2.0
endif
PYCONCAT=
ifneq ($(PYVERSION),)
	PYCONCAT=-
endif
PIP=pip
PYTHON=python
EASYINSTALL=easy_install
VIRTUALENV=virtualenv
SOURCE_ACTIVATE=. bin/activate; 


install-python27:
	$(call get,Python-2.7,Python-2.7.tgz,http://www.python.org/ftp/python/2.7)
	$(call compile,Python-2.7,,--prefix=/usr/local --with-threads --enable-shared)
	$(call download,setuptools-0.6c11-py2.7.egg,http://pypi.python.org/packages/2.7/s/setuptools)
	@ sudo sh setuptools-0.6c11-py2.7.egg

/usr/local/lib/$(LIBSQLITE3):
	$(call install,sqlite-autoconf-3080100,sqlite-autoconf-3080100.tar.gz,http://www.sqlite.org/2013)

sqlite3: /usr/local/lib/$(LIBSQLITE3)
	@ echo "[ setting up   ] sqlite3 database"
	@ cd imagedownloader/imagedownloader && cp -f database.sqlite3.py database.py

bin-postgres:
	$(call get,postgresql-9.2.4,postgresql-9.2.4.tar.gz,ftp://ftp.postgresql.org/pub/source/v9.2.4)
	$(call compile,postgresql-9.2.4,,--without-readline --without-zlib -with-python)

postgres: bin-postgres
	@ echo "[ setting up   ] postgres database"
	@ cd imagedownloader/imagedownloader && cp -f database.postgres.py database.py

/usr/local/lib/$(LIBHDF5):
	$(call get,hdf5-1.8.12,hdf5-1.8.12.tar.gz,http://www.hdfgroup.org/ftp/HDF5/current/src)
	$(call compile,hdf5-1.8.12,,--prefix=/usr/local --enable-shared --enable-hl)

/usr/local/lib/$(LIBNETCDF): /usr/local/lib/$(LIBHDF5)
	$(call get,netcdf-4.3.1-rc4,netcdf-4.3.1-rc4.tar.gz,ftp://ftp.unidata.ucar.edu/pub/netcdf)
	$(call compile,netcdf-4.3.1-rc4,LDFLAGS=-L/usr/local/lib CPPFLAGS=-I/usr/local/include,--enable-netcdf-4 --enable-dap --enable-shared --prefix=/usr/local)

imagedownloader/aspects.py:
	$(call get,python-aspects-1.3,python-aspects-1.3.tar.gz,http://www.cs.tut.fi/~ask/aspects)
	@ cp python-aspects-1.3/aspects.py imagedownloader/aspects.py

libs-and-headers: /usr/local/lib/$(LIBNETCDF) imagedownloader/aspects.py
	$(update_shared_libs)

bin/activate: imagedownloader/requirements.txt
	@ echo "[ installing   ] $(VIRTUALENV)"
	@ (sudo $(EASYINSTALL) virtualenv 2>&1) >> tracking.log
	@ echo "[ creating     ] $(VIRTUALENV) with no site packages"
	@ ($(VIRTUALENV) . --no-site-packages 2>&1) >> tracking.log
	@ echo "[ installing   ] $(PIP) inside $(VIRTUALENV)"
	@ ($(SOURCE_ACTIVATE) $(EASYINSTALL) pip 2>&1) >> tracking.log
	@ echo "[ installing   ] numpy inside $(VIRTUALENV)"
	@ ($(SOURCE_ACTIVATE) $(EASYINSTALL) numpy 2>&1) >> tracking.log
	@ echo "[ installing   ] $(PIP) requirements"
	@ ($(SOURCE_ACTIVATE) $(PIP) install -r imagedownloader/requirements.txt 2>&1) >> tracking.log
	@ touch bin/activate

db-migrate:
	@ echo "[ migrating    ] setting up the database structure"
	@ ($(SOURCE_ACTIVATE) cd imagedownloader && $(PYTHON) manage.py syncdb --noinput 2>&1) >> ../tracking.log
	@ ($(SOURCE_ACTIVATE) cd imagedownloader && $(PYTHON) manage.py migrate 2>&1) >> ../tracking.log

deploy: libs-and-headers bin/activate db-migrate

defaultsuperuser:
	@ echo "For the 'dev' user please select a password"
	@ $(SOURCE_ACTIVATE) cd imagedownloader && $(PYTHON) manage.py createsuperuser --username=dev --email=dev@dev.com

run:
	@ $(SOURCE_ACTIVATE) cd imagedownloader && $(PYTHON) manage.py runserver 8000

test:
	@ $(SOURCE_ACTIVATE) cd imagedownloader && $(PYTHON) manage.py test stations plumbing requester

test-coverage-travis-ci:
	@ $(SOURCE_ACTIVATE) cd imagedownloader && coverage run --source='stations/models.py,plumbing/models/*.py,requester/models.py' manage.py test stations plumbing requester

test-coveralls:
	@ $(SOURCE_ACTIVATE) cd imagedownloader && coveralls

test-coverage: test-coverage-travis-ci test-coveralls

clean:
	rm -rf sqlite* hdf5* netcdf-4* python-aspects* virtualenv* bin/ lib/ lib64 include/ build/ share Python-2.7* .Python ez_setup.py get-pip.py tracking.log imagedownloader/imagedownloader.sqlite3 imagedownloader/aspects.py
