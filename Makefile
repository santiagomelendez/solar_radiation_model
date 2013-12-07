OS:=$(shell uname -s)
download = [ ! -f $(1) ] && echo "[ downloading  ] $(1)" && curl -O $(2)/$(1) || echo "[ downloaded   ] $(1)"
unpack = [ ! -d $(2) ] && echo "[ unpacking    ] $(1)" && tar xzf $(1) || echo "[ unpacked     ] $(1)"

define get
	@ $(call download,$(2),$(3))
	@ $(call unpack,$(2),$(1))
endef

define compile
	@ cd $(1) && \
	([ -f ./configure ] && echo "[ configuring  ] $(1)" && $(2) sh ./configure $(3) >> tracking.log || echo "[ configured   ] $(1)") && \
	echo "[ compiling    ] $(1)" && \
	make -j 2 >> tracking.log && \
	echo "[ installing   ] $(1)" && \
	sudo make install >> tracking.log
endef

define install
	@ $(call get,$(1),$(2),$(3))
	@ $(call compile,$(1))
endef

PIP=pip
PYTHON=python
update_shared_libs=sudo ldconfig

ifeq ($(OS), Darwin)
	update_shared_libs=
	PIP=pip-2.7
	PYTHON=python2.7
else
	DISTRO=$(shell lsb_release -si)
	ifeq ($(DISTRO), CentOS)
		PIP=pip-2.7
		PYTHON=python2.7
	endif
endif

test:
	@ cd imagedownloader && $(PYTHON) manage.py test stations

run:
	@ cd imagedownloader && $(PYTHON) manage.py runserver 8000

defaultsuperuser:
	@ echo "For the 'dev' user please select a password"
	@ cd imagedownloader && $(PYTHON) manage.py createsuperuser --username=dev --email=dev@dev.com

db-migrate:
	@ echo "[ migrating    ]"
	@ cd imagedownloader && $(PYTHON) manage.py syncdb --noinput
	@ cd imagedownloader && $(PYTHON) manage.py migrate

deploy-withoutpip: pip-requirements db-migrate

deploy: bin-pip deploy-withoutpip

bin-sqlite3:
	$(call install,sqlite-autoconf-3080100,sqlite-autoconf-3080100.tar.gz,http://www.sqlite.org/2013)

src-python27:
	$(call install,Python-2.7,Python-2.7.tgz,http://www.python.org/ftp/python/2.7)

lib-hdf5:
	$(call get,hdf5-1.8.12,hdf5-1.8.12.tar.gz,http://www.hdfgroup.org/ftp/HDF5/current/src)
	$(call compile,hdf5-1.8.12,,--prefix=/usr/local --enable-shared --enable-hl)

lib-netcdf4: lib-hdf5
	$(call get,netcdf-4.3.1-rc4,netcdf-4.3.1-rc4.tar.gz,ftp://ftp.unidata.ucar.edu/pub/netcdf)
	$(call compile,netcdf-4.3.1-rc4,LDFLAGS=-L/usr/local/lib CPPFLAGS=-I/usr/local/include,--enable-netcdf-4 --enable-dap --enable-shared --prefix=/usr/local)

src-aspects:
	$(call install,python-aspects-1.3,python-aspects-1.3.tar.gz,http://www.cs.tut.fi/~ask/aspects)
	@ sudo cp python-aspects-1.3/aspects.py imagedownloader/aspects.py

src-postgres:
	--without-readline --without-zlib

bin-pip:
	@ $(call download,ez_setup.py,https://bitbucket.org/pypa/setuptools/raw/bootstrap)
	@ echo "[ installing   ] $(PYTHON) setuptools"
	@ sudo $(PYTHON) ez_setup.py 
	@ $(call download,get-pip.py,https://raw.github.com/pypa/pip/master/contrib)
	@ echo "[ installing   ] $(PIP)"
	@ sudo $(PYTHON) get-pip.py

postgres:
	@ echo "[ setting up   ] postgres database"
	@ cd imagedownloader && cp -f databsase.postgres.py database.py

sqlite3: bin-sqlite3
	@ echo "[ setting up   ] sqlite3 database"
	@ cd imagedownloader/imagedownloader && cp -f database.sqlite3.py database.py

pip-requirements: lib-netcdf4 src-aspects
	@ $(update_shared_libs)
	@ echo "[ installing   ] $(PIP) requirements"
	@ $(PIP) install -r imagedownloader/requirements.txt --upgrade

clean:
	sudo rm -rf sqlite* hdf5* netcdf-4* python-aspects* ez_setup.py get-pip.py tracking.log imagedownloader/imagedownloader.sqlite3
