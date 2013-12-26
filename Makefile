OS:=$(shell uname -s)
download = [ ! -f $(1) ] && echo "[ downloading  ] $(1)" && curl -O $(2)/$(1) || echo "[ downloaded   ] $(1)"
unpack = [ ! -d $(2) ] && echo "[ unpacking    ] $(1)" && tar xzf $(1) || echo "[ unpacked     ] $(1)"

define get
	@ $(call download,$(2),$(3))
	@ $(call unpack,$(2),$(1))
endef

define compile
	@ cd $(1) && \
	([ -f ./configure ] && echo "[ configuring  ] $(1)" && ($(2) sh ./configure $(3) 2>&1) >> tracking.log || echo "[ configured   ] $(1)") && \
	echo "[ compiling    ] $(1)" && \
	(make -j 2 2>&1) >> tracking.log && \
	echo "[ installing   ] $(1)" && \
	(sudo make install 2>&1) >> tracking.log
endef

define install
	@ $(call get,$(1),$(2),$(3))
	@ $(call compile,$(1))
endef

update_shared_libs=sudo ldconfig

ifeq ($(OS), Darwin)
	update_shared_libs=
	#else
	#DISTRO=$(shell lsb_release -si)
	#ifeq ($(DISTRO), CentOS)
	#endif
endif
PYCONCAT=
ifneq ($(PYVERSION),)
	PYCONCAT=-
endif
PIP=pip$(PYCONCAT)$(PYVERSION)
PYTHON=python$(PYVERSION)
EASYINSTALL=easy_install$(PYCONCAT)$(PYVERSION)
VIRTUALENV=virtualenv$(PYCONCAT)$(PYVERSION)
SOURCE_ACTIVATE=. bin/activate;


show-versions:
	@ $(shell $(PYTHON) --version)

test:
	@ $(SOURCE_ACTIVATE) cd imagedownloader && $(PYTHON) manage.py test stations plumbing requester

test-coverage-travis-ci:
	@ $(SOURCE_ACTIVATE) cd imagedownloader && coverage run --source='stations/models.py,plumbing/models/*.py,requester/models.py' manage.py test stations plumbing requester

test-coveralls:
	. $(SOURCE_ACTIVATE) cd imagedownloader && coveralls

test-coverage: test-coverage-travis-ci test-coveralls

run:
	@ $(SOURCE_ACTIVATE) cd imagedownloader && $(PYTHON) manage.py runserver 8000

defaultsuperuser:
	@ echo "For the 'dev' user please select a password"
	@ $(SOURCE_ACTIVATE) cd imagedownloader && $(PYTHON) manage.py createsuperuser --username=dev --email=dev@dev.com

db-migrate:
	@ echo "[ migrating    ] setting up the database structure"
	@ ($(SOURCE_ACTIVATE) cd imagedownloader && $(PYTHON) manage.py syncdb --noinput 2>&1) >> tracking.log
	@ ($(SOURCE_ACTIVATE) cd imagedownloader && $(PYTHON) manage.py migrate 2>&1) >> tracking.log

libs-and-headers: lib-netcdf4 src-aspects
	$(update_shared_libs)

deploy-withoutpip: libs-and-headers bin/activate db-migrate

deploy: deploy-withoutpip

bin-sqlite3:
	$(call install,sqlite-autoconf-3080100,sqlite-autoconf-3080100.tar.gz,http://www.sqlite.org/2013)

install-python27:
	$(call install,Python-2.7,Python-2.7.tgz,http://www.python.org/ftp/python/2.7)

lib-hdf5:
	$(call get,hdf5-1.8.12,hdf5-1.8.12.tar.gz,http://www.hdfgroup.org/ftp/HDF5/current/src)
	$(call compile,hdf5-1.8.12,,--prefix=/usr/local --enable-shared --enable-hl)

lib-netcdf4: lib-hdf5
	$(call get,netcdf-4.3.1-rc4,netcdf-4.3.1-rc4.tar.gz,ftp://ftp.unidata.ucar.edu/pub/netcdf)
	$(call compile,netcdf-4.3.1-rc4,LDFLAGS=-L/usr/local/lib CPPFLAGS=-I/usr/local/include,--enable-netcdf-4 --enable-dap --enable-shared --prefix=/usr/local)

src-aspects:
	$(call get,python-aspects-1.3,python-aspects-1.3.tar.gz,http://www.cs.tut.fi/~ask/aspects)
	@ cp python-aspects-1.3/aspects.py imagedownloader/aspects.py

src-postgres:
	$(call get,postgresql-9.2.4,postgresql-9.2.4.tar.gz,ftp://ftp.postgresql.org/pub/source/v9.2.4)
	$(call compile,postgresql-9.2.4,,--without-readline --without-zlib -with-python)

postgres:
	@ echo "[ setting up   ] postgres database"
	@ cd imagedownloader/imagedownloader && cp -f database.postgres.py database.py

sqlite3: bin-sqlite3
	@ echo "[ setting up   ] sqlite3 database"
	@ cd imagedownloader/imagedownloader && cp -f database.sqlite3.py database.py

bin/activate: show-versions imagedownloader/requirements.txt
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

clean:
	rm -rf sqlite* hdf5* netcdf-4* python-aspects* virtualenv* bin/ lib/ include/ build/ share Python-2.7* .Python ez_setup.py get-pip.py tracking.log imagedownloader/imagedownloader.sqlite3 imagedownloader/aspects.py