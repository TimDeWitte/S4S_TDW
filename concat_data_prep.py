# # Concatenated data preparation
#
# Prepare data of multiple runs for model calibration
#
# * Clean data using standard process
# * Manual adjustement of probe state (inserted in track or not) in some cases
# * Concatenate subsequent runs
# * Save prepared data to pickle files (containing `RunData` objects)

# +
import importlib
imported_module = importlib.import_module("data_utils")
importlib.reload(imported_module)
from data_utils import *


import data_utils
import numpy as np
import pandas
import matplotlib
from matplotlib import pyplot as plt
from pathlib import Path
import pickle

plt.style.use("ggplot")

# %matplotlib widget

datadir = Path(
    r"C:\Users\TDewitte\Desktop\PythonPrograms\S4S\camso-s4s\tdms_data"
)

# Directory where the .pickle files containing processed data will be created
output_dir = Path("./concat_data")

# Create directory if it doesn't exist
output_dir.mkdir(exist_ok=True)


# -

def plot_tms(rdata):
    fig, axes = plt.subplots(4, 2, figsize=[8, 12])
    axrow, axcol = 0, 0
    for corner in ["RR", "FR", "RL", "RL"]:
        axcol = 0
        for board in ["in", "out"]:
            ax = axes[axrow, axcol]
            tms = rdata.get_tms_channels(corner, board)
            tms.plot(ax=ax)
            ax.set(title=f"{corner} {board}board")
            axcol += 1
        axrow += 1


# ## Runs 29 - 31
#
# Corresponding to test sequences 39, 40, 42
#
# Load: Heavy
#
# Speed: 25 km/h
#
# Condition: Flat, straddle, crown

data_run29_raw = data_utils.load_run(datadir, 29)
data_run29_clean = data_utils.clean_data(data_run29_raw, "out")

# +
data_run29_clean.probe.plot()

data_run29_clean.probe_state.loc["2022-08-29 14:12:30":"2022-08-29 14:25:30"] = "out"
data_run29_clean.probe_state.loc[:"2022-08-29 13:38:30"] = "in"

data_run29_clean.probe[data_run29_clean.probe_state == "in"].plot()

# +
# Ambient temperature estimate

fig, ax = plt.subplots()
a = data_run29_clean.probe.min(axis=1)
a[(data_run29_clean.probe_state != "out").any(axis=1)] = np.nan
ax.plot(a)

a = a.fillna(method="ffill")
ax.plot(a)

med = a.rolling(60*10, center=True).median().fillna(method="ffill").fillna(method="bfill")
ax.plot(med, color="black")
# -

data_run29_clean.malt.plot()

data_run29_clean.smart_track.plot()

data_run29_clean.can.plot()

plot_tms(data_run29_clean)

data_run30_raw = data_utils.load_run(datadir, 30)

data_run30_clean = data_utils.clean_data(data_run30_raw, "in")

# +
data_run30_raw.probe.plot()

# Front channels appear to be dead
data_run30_clean.probe.loc[:, ["probe_FR_IN", "probe_FR_OUT", "probe_FL_IN", "probe_FL_OUT"]] = np.nan

data_run30_clean.probe.plot()
data_run30_clean.probe[data_run30_clean.probe_state == "in"].plot()
# -

data_run30_clean.smart_track.plot()

data_run30_clean.malt.plot()

data_run30_clean.can.plot()

data_run31_raw = data_utils.load_run(datadir, 31)
data_run31_clean = data_utils.clean_data(data_run31_raw, "in")

# +
data_run31_raw.probe.plot()

# Front channels appear to be dead
data_run31_clean.probe.loc[:, ["probe_FR_IN", "probe_FR_OUT", "probe_FL_IN", "probe_FL_OUT"]] = np.nan

data_run31_clean.probe.plot()
data_run31_clean.probe[data_run31_clean.probe_state == "in"].plot()
# -

data_run31_clean.smart_track.plot()
data_run31_clean.can.plot()
data_run31_clean.malt.plot()

# +
cdata_29 = data_utils.concat_data([data_run29_clean, data_run30_clean, data_run31_clean])
with open(output_dir / "runs_29_30_31.pickle", "wb") as f:
    pickle.dump(cdata_29, f)
    
ax = cdata_29.probe.plot()
ax.plot(cdata_29.get_ambient_est(), color="black")
cdata_29.probe[cdata_29.probe_state == "in"].plot()

cdata_29.smart_track.plot()
cdata_29.can.plot()
cdata_29.malt.plot()

plot_tms(cdata_29)
# -

# ## Runs 5, 6, 7
#
# Corresponding to test sequences 6, 7, 8
#
# Load: Unladen
#
# Speed: 25 km/h
#
# Condition: Cool-down, flat, straddle

data_run5_raw = data_utils.load_run(datadir, 5)
data_run5_clean = data_utils.clean_data(data_run5_raw, "in")

data_run5_raw.probe.plot()
ax = data_run5_clean.probe.plot()
#ax.plot(data_run5_clean.get_ambient_est(), color="k", label="ambient")
#ax.legend()
data_run5_clean.get_probe_in().plot()

data_run5_clean.malt.plot()
data_run5_clean.smart_track.plot()
data_run5_clean.can.plot()

data_run6_raw = data_utils.load_run(datadir, 6)
data_run6_clean = data_utils.clean_data(data_run6_raw, "out")
plt.close("all")

data_run6_raw.probe.plot()
ax = data_run6_clean.probe.plot()
data_run6_clean.get_ambient_est().plot(ax=ax)

data_run6_clean.malt.plot()
data_run6_clean.smart_track.plot()
data_run6_clean.can.plot()

data_run7_raw = data_utils.load_run(datadir, 7)
data_run7_clean = data_utils.clean_data(data_run7_raw, "out")
plt.close("all")

# +
data_run7_raw.probe.plot()

# RR_IN probe channel is spiky, use median filter to remove spikes
data_run7_clean.probe["probe_RR_IN"] = (
    data_run7_clean.probe["probe_RR_IN"]
    .rolling(30, center=True)
    .median()
    .interpolate()
    .fillna(method="bfill")
)
print(data_run7_clean.probe.info())

ax = data_run7_clean.probe.plot()
data_run7_clean.get_ambient_est().plot(ax=ax)

# +
cdata_5 = data_utils.concat_data([data_run5_clean, data_run6_clean, data_run7_clean])
with open(output_dir / "runs_5_6_7.pickle", "wb") as f:
    pickle.dump(cdata_5, f)
    
ax = cdata_5.probe.plot()
ax.plot(cdata_5.get_ambient_est(), color="black")
cdata_5.get_probe_in().plot()

cdata_5.smart_track.plot()
cdata_5.can.plot()
cdata_5.malt.plot()
plot_tms(cdata_5)
# -

t1 = cdata_5.get_probe_in()["probe_RR_OUT"]
t1.head()

# ## Runs 18, 19, 20
#
# Corresponding to test sequences 23, 27, 29
#
# Load: Light
#
# Speed: 25 km/h, 32.5 km/h
#
# Condition: Crown, straddle, crown
#
# Morning cold start for run 18 (all temperatures at ambient)

data_run18_raw = data_utils.load_run(datadir, 18)
data_run19_raw = data_utils.load_run(datadir, 19)
data_run20_raw = data_utils.load_run(datadir, 20)
#data_run21_raw = data_utils.load_run(datadir, 21)

data_run18_clean = data_utils.clean_data(data_run18_raw, "out")

# +
data_run18_raw.probe.plot()
data_run18_clean.probe.plot()
ax = data_run18_clean.get_probe_in().plot()
data_run18_clean.get_ambient_est().plot(ax=ax, color="black")

data_run18_clean.malt.plot()
data_run18_clean.smart_track.plot()
data_run18_clean.can.plot()

# +
data_run19_clean = data_utils.clean_data(data_run19_raw, "out")

data_run19_raw.probe.plot()
data_run19_clean.probe.plot()
ax = data_run19_clean.get_probe_in().plot()
data_run19_clean.get_ambient_est().plot(ax=ax, color="black")

data_run19_clean.malt.plot()
data_run19_clean.smart_track.plot()
data_run19_clean.can.plot()

# +
data_run20_clean = data_utils.clean_data(data_run20_raw, "out")

data_run20_raw.probe.plot()
data_run20_clean.probe.plot()
ax = data_run20_clean.get_probe_in().plot()
data_run20_clean.get_ambient_est().plot(ax=ax, color="black")

data_run20_clean.malt.plot()
data_run20_clean.smart_track.plot()
data_run20_clean.can.plot()

# +
cdata_18 = data_utils.concat_data([data_run18_clean, data_run19_clean, data_run20_clean])
with open(output_dir / "runs_18_19_20.pickle", "wb") as f:
    pickle.dump(cdata_18, f)
    
ax = cdata_18.probe.plot()
ax.plot(cdata_18.get_ambient_est(), color="black")
cdata_18.get_probe_in().plot()

cdata_18.smart_track.plot()
cdata_18.can.plot()
cdata_18.malt.plot()
plot_tms(cdata_18)
# -
cdata_18.probe.index




# ## Runs 1, 2, 3, 4
#
# Corresponding to test sequences 1, 2, 3, 4
#
# Load: ??
#
# Speed: ??
#
# Condition: ??
#
# ??

data_run1_raw = data_utils.load_run(datadir, 1)
data_run1_clean = data_utils.clean_data(data_run1_raw, "out")

# +

data_run1_raw.probe.plot()
data_run1_clean.probe.plot()
ax = data_run1_clean.get_probe_in().plot()
data_run1_clean.get_ambient_est().plot(ax=ax, color="black")

data_run1_clean.malt.plot()
data_run1_clean.smart_track.plot()
data_run1_clean.can.plot()


# +
data_run2_raw = data_utils.load_run(datadir, 2)
data_run2_clean = data_utils.clean_data(data_run2_raw, "out")

data_run2_raw.probe.plot()
data_run2_clean.probe.plot()
ax = data_run2_clean.get_probe_in().plot()
data_run2_clean.get_ambient_est().plot(ax=ax, color="black")

data_run2_clean.malt.plot()
data_run2_clean.smart_track.plot()
data_run2_clean.can.plot()


# +
data_run3_raw = data_utils.load_run(datadir, 3)
data_run3_clean = data_utils.clean_data(data_run3_raw, "out")

data_run3_raw.probe.plot()
data_run3_clean.probe.plot()
ax = data_run3_clean.get_probe_in().plot()
data_run3_clean.get_ambient_est().plot(ax=ax, color="black")

data_run3_clean.malt.plot()
data_run3_clean.smart_track.plot()
data_run3_clean.can.plot()


# +
data_run4_raw = data_utils.load_run(datadir, 4)
data_run4_clean = data_utils.clean_data(data_run4_raw, "out")

data_run4_raw.probe.plot()
data_run4_clean.probe.plot()
ax = data_run4_clean.get_probe_in().plot()
data_run4_clean.get_ambient_est().plot(ax=ax, color="black")

data_run4_clean.malt.plot()
data_run4_clean.smart_track.plot()
data_run4_clean.can.plot()


# + active=""
# data_run5_raw = data_utils.load_run(datadir, 5)
# data_run5_clean = data_utils.clean_data(data_run5_raw, "out")
# """
# data_run5_raw.probe.plot()
# data_run5_clean.probe.plot()
# ax = data_run5_clean.get_probe_in().plot()
# data_run5_clean.get_ambient_est().plot(ax=ax, color="black")
#
# data_run5_clean.malt.plot()
# data_run5_clean.smart_track.plot()
# data_run5_clean.can.plot()
# """

# +
cdata_14 = data_utils.concat_data([data_run1_clean, data_run2_clean, data_run3_clean, data_run4_clean])
with open(output_dir / "runs_1_2_3_4.pickle", "wb") as f:
    pickle.dump(cdata_14, f)
    
ax = cdata_14.probe.plot()
ax.plot(cdata_14.get_ambient_est(), color="black")
cdata_14.get_probe_in().plot()

cdata_14.smart_track.plot()
cdata_14.can.plot()
cdata_14.malt.plot()
plot_tms(cdata_14)
# -

# ## Runs 8, 9, 10
#
# Corresponding to test sequences 8, 9, 10
#
# Load: ??
#
# Speed: ??
#
# Condition: ??
#
# ??

data_run8_raw = data_utils.load_run(datadir, 8)
data_run8_clean = data_utils.clean_data(data_run8_raw, "out")

data_run9_raw = data_utils.load_run(datadir, 9)
data_run9_clean = data_utils.clean_data(data_run9_raw, "out")

data_run10_raw = data_utils.load_run(datadir, 10)
data_run10_clean = data_utils.clean_data(data_run10_raw, "out")

# +
cdata_810 = data_utils.concat_data([data_run8_clean, data_run9_clean, data_run10_clean])
with open(output_dir / "runs_8_9_10.pickle", "wb") as f:
    pickle.dump(cdata_810, f)
    
ax = cdata_810.probe.plot()
ax.plot(cdata_810.get_ambient_est(), color="black")
cdata_810.get_probe_in().plot()

cdata_810.smart_track.plot()
cdata_810.can.plot()
cdata_810.malt.plot()
plot_tms(cdata_810)
# -

# ## Runs 11, 12, 13, 14
#
# Corresponding to test sequences 11, 12, 13, 14
#
# Load: ??
#
# Speed: ??
#
# Condition: ??
#
# ??

data_run11_raw = data_utils.load_run(datadir, 11)
data_run11_clean = data_utils.clean_data(data_run11_raw, "out")

data_run12_raw = data_utils.load_run(datadir, 12)
data_run12_clean = data_utils.clean_data(data_run12_raw, "out")

data_run13_raw = data_utils.load_run(datadir, 13)
data_run13_clean = data_utils.clean_data(data_run13_raw, "out")

data_run14_raw = data_utils.load_run(datadir, 14)
data_run14_clean = data_utils.clean_data(data_run14_raw, "out")

# +
cdata_1114 = data_utils.concat_data([data_run11_clean, data_run12_clean, data_run13_clean, data_run14_clean])
with open(output_dir / "runs_11_12_13_14.pickle", "wb") as f:
    pickle.dump(cdata_1114, f)
    
ax = cdata_1114.probe.plot()
ax.plot(cdata_1114.get_ambient_est(), color="black")
cdata_1114.get_probe_in().plot()

cdata_1114.smart_track.plot()
cdata_1114.can.plot()
cdata_1114.malt.plot()
plot_tms(cdata_1114)
# -

# ## Runs 15, 16, 17
#
# Corresponding to test sequences 15, 16, 17
#
# Load: ??
#
# Speed: ??
#
# Condition: ??
#
# ??

data_run15_raw = data_utils.load_run(datadir, 15)
data_run15_clean = data_utils.clean_data(data_run15_raw, "out")

data_run16_raw = data_utils.load_run(datadir, 16)
data_run16_clean = data_utils.clean_data(data_run16_raw, "out")

data_run17_raw = data_utils.load_run(datadir, 17)
data_run17_clean = data_utils.clean_data(data_run17_raw, "out")

# +
cdata_1517 = data_utils.concat_data([data_run15_clean, data_run16_clean, data_run17_clean])
with open(output_dir / "runs_15_16_17.pickle", "wb") as f:
    pickle.dump(cdata_1517, f)
    
ax = cdata_1517.probe.plot()
ax.plot(cdata_1517.get_ambient_est(), color="black")
cdata_1517.get_probe_in().plot()

cdata_1517.smart_track.plot()
cdata_1517.can.plot()
cdata_1517.malt.plot()
plot_tms(cdata_1517)
# -

# ## Runs 18-28
#
# Corresponding to test sequences 18-28
#
# Load: ??
#
# Speed: ??
#
# Condition: ??
#
# ??

data_run21_raw = data_utils.load_run(datadir, 21)
data_run21_clean = data_utils.clean_data(data_run21_raw, "out")

data_run22_raw = data_utils.load_run(datadir, 22)
data_run22_clean = data_utils.clean_data(data_run22_raw, "out")

data_run23_raw = data_utils.load_run(datadir, 23)
data_run23_clean = data_utils.clean_data(data_run23_raw, "out")

data_run24_raw = data_utils.load_run(datadir, 24)
data_run24_clean = data_utils.clean_data(data_run24_raw, "out")

data_run25_raw = data_utils.load_run(datadir, 25)
data_run25_clean = data_utils.clean_data(data_run25_raw, "out")

data_run26_raw = data_utils.load_run(datadir, 26)
data_run26_clean = data_utils.clean_data(data_run26_raw, "out")

data_run27_raw = data_utils.load_run(datadir, 27)
data_run27_clean = data_utils.clean_data(data_run27_raw, "out")

data_run28_raw = data_utils.load_run(datadir, 28)
data_run28_clean = data_utils.clean_data(data_run28_raw, "out")

# +
cdata_1828 = data_utils.concat_data([data_run18_clean, data_run19_clean, data_run20_clean, data_run21_clean, data_run22_clean, data_run23_clean, data_run24_clean, data_run25_clean, data_run26_clean, data_run27_clean, data_run28_clean])
with open(output_dir / "runs_18-28.pickle", "wb") as f:
    pickle.dump(cdata_1828, f)
    
ax = cdata_1828.probe.plot()
ax.plot(cdata_1828.get_ambient_est(), color="black")
cdata_1828.get_probe_in().plot()

cdata_1828.smart_track.plot()
cdata_1828.can.plot()
cdata_1828.malt.plot()
plot_tms(cdata_1828)
# -

# ### Runs 32-34

data_run32_raw = data_utils.load_run(datadir, 32)
data_run32_clean = data_utils.clean_data(data_run32_raw, "out")

data_run33_raw = data_utils.load_run(datadir, 33)
data_run33_clean = data_utils.clean_data(data_run33_raw, "out")

data_run34_raw = data_utils.load_run(datadir, 34)
data_run34_clean = data_utils.clean_data(data_run34_raw, "out")

# +
cdata_3234 = data_utils.concat_data([data_run32_clean, data_run33_clean, data_run34_clean])
with open(output_dir / "runs_32_33_34.pickle", "wb") as f:
    pickle.dump(cdata_3234, f)
    
ax = cdata_3234.probe.plot()
ax.plot(cdata_3234.get_ambient_est(), color="black")
cdata_3234.get_probe_in().plot()

cdata_3234.smart_track.plot()
cdata_3234.can.plot()
cdata_3234.malt.plot()
plot_tms(cdata_3234)
# -
# ## All runs 

# +
cdata_all = data_utils.concat_data([data_run1_clean, data_run2_clean, data_run3_clean, data_run4_clean, data_run5_clean, data_run6_clean, data_run7_clean, data_run8_clean,
                                   data_run9_clean, data_run10_clean, data_run11_clean, data_run12_clean, data_run13_clean, data_run14_clean, data_run15_clean, data_run16_clean,
                                   data_run17_clean, data_run18_clean, data_run19_clean, data_run20_clean, data_run21_clean, data_run22_clean, data_run23_clean, data_run24_clean,
                                   data_run25_clean, data_run26_clean, data_run27_clean, data_run28_clean, data_run29_clean, data_run30_clean, data_run31_clean, data_run32_clean,
                                   data_run33_clean, data_run34_clean])
with open(output_dir / "runs_all.pickle", "wb") as f:
    pickle.dump(cdata_all, f)
    
ax = cdata_all.probe.plot()
ax.plot(cdata_all.get_ambient_est(), color="black")
cdata_all.get_probe_in().plot()

cdata_all.smart_track.plot()
cdata_all.can.plot()
cdata_all.malt.plot()
plot_tms(cdata_all)

