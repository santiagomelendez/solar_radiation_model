import sys
from datetime import datetime
import pytz


def total_seconds(td):
	return (td.microseconds + (td.seconds + td.days * 24 * 3600) * 10**6) / 10**6

def show(*objs):
	begin = '' if '\r' in objs[0] or '\b' in objs[0] else '\n'
	sys.stdout.write(begin)
	for part in objs:
		sys.stdout.write(str(part))
	sys.stdout.flush()

def say(speech):
	show(speech)

progress = ['/','-','\\','|']
def show_progress(i):
	show('\b \b', progress[i % len(progress)])
