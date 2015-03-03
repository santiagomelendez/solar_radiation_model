import heliosat
from goesdownloader import instrument as goes


def run():
    filenames = goes.download('noaa.gvarim', 'noaaadmin', 'data_new',
                              suscription_id='55253')
    if filenames:
        heliosat.workwith('data/goes13.*.BAND_01.nc')
