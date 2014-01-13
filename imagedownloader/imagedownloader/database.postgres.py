DATABASES = {}
DATABASES = {
	'default': {
		#'ENGINE': 'django.db.backends.sqlite3',
		#'NAME': 'filedb.sqlite3',
	'ENGINE': 'django.db.backends.postgresql_psycopg2',
	'NAME': 'imagedownloader',
	'USER': 'postgres',
	'PASSWORD': 'postgres',
	'HOST': 'localhost',
	'PORT': '5432',
	}
}