import heliosat2
import glob
from goesdownloader import instrument as goes


def run():
    should_download = lambda dt: dt.hour - 4 >= 5 and dt.hour - 4 <= 20
    filenames = goes.download('user', 'pass', 'data', suscription_id='',
                              datetime_filter=should_download)
    if filenames:
        heliosat2.workwith('data/goes13.*.BAND_01.nc')
