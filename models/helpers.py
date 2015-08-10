import sys
sys.path.append(".")
from datetime import datetime


def short(f, start=1, end=-2):
    return ".".join((f.split('/')[-1]).split('.')[start:end])


def to_datetime(filename):
    return datetime.strptime(short(filename), '%Y.%j.%H%M%S')

