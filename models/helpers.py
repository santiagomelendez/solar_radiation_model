import sys
sys.path.append(".")
from datetime import datetime


def show(*objs):
    begin = '' if '\r' in objs[0] or '\b' in objs[0] else '\n'
    sys.stdout.write(begin)
    for part in objs:
        sys.stdout.write(str(part))
    sys.stdout.flush()


def short(f, start=1, end=-2):
    return ".".join((f.split('/')[-1]).split('.')[start:end])


def to_datetime(filename):
    return datetime.strptime(short(filename), '%Y.%j.%H%M%S')

