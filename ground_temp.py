# +
from pathlib import Path

import numpy as np

import pandas

from nptdms import TdmsFile

from matplotlib import pyplot as plt

import data_utils

plt.style.use("ggplot")

# %matplotlib widget

datadir = Path(
    r"\\meca.gme.usherbrooke.ca\CREATEK\Contrats\Michelin S4S\Contrat 2\Michelin test data"
)


# -

def load_ground_temp(datadir, runs):
    df_list = []
    for run in runs:
        datafile = next(datadir.glob(f"DT-3105-RUN{run}_*.tdms"))
        with TdmsFile.open(datafile) as f:
            can_grtemp_chan = [c for c in f["Group4"] if c.startswith("IR_7_1@GR_Temp")][0]
            grtemp = f["Group4"][can_grtemp_chan][:]
            time = f["Group4"]["Time"][:]
            timestamps = data_utils.get_dtindex_from_tdms(f, time)
            df = pandas.DataFrame(
                {"timestamp": timestamps, "GroundTemp": grtemp}
            ).set_index("timestamp", verify_integrity=True)
            df = data_utils.lpfilt_df(df, fc=0.05)
            df_list.append(df)

    return pandas.concat(df_list, axis=0).sort_index()


gtemp_5 = load_ground_temp(datadir, [5, 6, 7])
gtemp_18 = load_ground_temp(datadir, [18, 19, 20])
gtemp_29 = load_ground_temp(datadir, [29, 30, 31])

gtemp_5.plot()
gtemp_18.plot()
gtemp_29.plot()

gtemp_18.head()


