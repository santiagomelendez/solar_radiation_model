from core import Loader, show
from heliosat2 import Heliosat2

def workwith(filename="data/goes13.*.BAND_01.nc"):
    filenames = filter_filenames(filename)
    months = list(set(map(lambda dt: '%i/%i' % (dt.month, dt.year),
                          map(to_datetime, filenames))))
    show("=======================")
    show("Months: ", months)
    show("Dataset: ", len(filenames), " files.")
    show("-----------------------\n")
    loader = Loader(filenames)
    strategy = Heliosat2(filenames)
    strategy.run_with(loader)
    show("Process finished.\n")
    print loader.freq
