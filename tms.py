# +
import numpy as np

import pandas

from matplotlib import pyplot as plt

import matplotlib

import pickle

import data_utils

import kalman

plt.style.use("ggplot")

# #%matplotlib widget
# +
with open("./concat_data/runs_5_6_7.pickle", "rb") as f:
    rdata_5_6_7 = pickle.load(f)
    
with open("./concat_data/runs_29_30_31.pickle", "rb") as f:
    rdata_29_30_31 = pickle.load(f)
    
with open("./concat_data/runs_18_19_20.pickle", "rb") as f:
    rdata_18_19_20 = pickle.load(f)

with open("./concat_data/runs_1_2_3_4.pickle", "rb") as f:
    rdata_1_2_3_4 = pickle.load(f)

with open("./concat_data/runs_8_9_10.pickle", "rb") as f:
    rdata_8_9_10 = pickle.load(f)

with open("./concat_data/runs_11_12_13_14.pickle", "rb") as f:
    rdata_11_12_13_14 = pickle.load(f)

with open("./concat_data/runs_15_16_17.pickle", "rb") as f:
    rdata_15_16_17 = pickle.load(f)

    
all_run_datasets = {
    "5, 6, 7": rdata_5_6_7,
    "18, 19, 20": rdata_18_19_20,
    "29, 30, 31": rdata_29_30_31,
    "1, 2, 3, 4": rdata_1_2_3_4,
    "8, 9, 10": rdata_8_9_10,
    "11, 12, 13, 14": rdata_11_12_13_14,
    "15, 16, 17": rdata_15_16_17
}


# +
# %matplotlib tk

rdata = all_run_datasets["29, 30, 31"]
fig, axes = plt.subplots(4, 2, figsize=[8, 12])
fig.tight_layout(h_pad=10)
axrow, axcol = 0, 0
for corner in ["FL", "FR", "RL", "RR"]:
    axcol = 0
    for board in ["in", "out"]:
        ax = axes[axrow, axcol]
        probe = rdata.get_probe_in(corner, board)
        ir = rdata.get_ir_temp(corner, board)
        tms = rdata.get_tms_channels(corner, board, mean=False)
        tms_mean = rdata.get_tms_channels(corner=corner, inout=board, mean=1)
        tms_median = rdata.get_tms_channels(corner=corner, inout=board, mean=2)
        tms.plot(ax=ax)
        ax.plot(tms_mean, label="TMS (mean)", color="C2", linewidth=3.0)
        ax.plot(tms_median, label="TMS (median)", color="C4", linewidth=3.0)
        ax.plot(probe, label="Probe", color="C5", linewidth=3.0)
        ax.plot(ir, label="IR", color="C6")
        ax.set(title=f"{corner} {board}board")
        ax.set_ylim([10, 160])
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.5), ncol=3)
        axcol += 1
    
    axrow += 1

# +
# %matplotlib widget

"""1 RR in"""

fig, axes = plt.subplots(7, 1, figsize=[8, 20])  # the same number of plots as you have groups of runs (all_run_datasets)

i = 0
for runs_str, rdata in all_run_datasets.items():
    #rdata.tms.info()
    
    ax = axes[i]
    #rdata.smart_track.plot(ax=ax)
    ax.plot(rdata.smart_track["RR"], linewidth=3.0, label="Smart Track")

    ax.plot(rdata.get_ir_temp(corner="RR", inout="in"), color="C6",label="IR")

    ax.plot(rdata.can["GroundTemp"], color="C7",label="Ambient")
    
    ax.plot(rdata.get_probe_in(corner="RR", inout="in"), linewidth=3.0, color="C5",label="Probe")
    
    tms_cols = [c for c in rdata.tms.columns if c.startswith("RR_in")]
    
    tms_mean = rdata.get_tms_channels(corner="RR", inout="in", mean=1)
    tms_median = rdata.get_tms_channels(corner="RR", inout="in", mean=2)
    ax.plot(tms_mean, label="TMS (mean)", color="C2", linewidth=3.0)
    ax.plot(tms_median, label="TMS (median)", color="C4", linewidth=3.0)

    for col in tms_cols:
        line = ax.plot(rdata.tms[col], color="C1", linewidth=1.0, alpha=0.5)
    line[0].set(label="TMS")
    
    ax.set(ylabel="Temperature (°C)")
    ax.set_ylim([20, 80])
    
    # ax.legend()  # If you want a legend to be shown (over the graph, makes it unclear)

    i += 1
# -
for runs_str, rdata in all_run_datasets.items():
    #rdata.tms.info()
    print(type(rdata))

# +
"""2 RR out"""

fig, axes = plt.subplots(7, 1, figsize=[8, 11])  # the same number of plots as you have groups of runs (all_run_datasets)

i = 0
for runs_str, rdata in all_run_datasets.items():
    #rdata.tms.info()
    
    ax = axes[i]
    #rdata.smart_track.plot(ax=ax)
    ax.plot(rdata.smart_track["RR"], linewidth=3.0, label="Smart Track")

    ax.plot(rdata.get_ir_temp(corner="RR", inout="in"), color="C6",label="Probe")

    ax.plot(rdata.get_probe_in(corner="RR", inout="in"), linewidth=3.0, color="C5",label="Probe")
        
    tms_cols = [c for c in rdata.tms.columns if c.startswith("RR_out")]
    
    tms_mean = rdata.get_tms_channels(corner="RR", inout="out", mean=1)
    tms_median = rdata.get_tms_channels(corner="RR", inout="out", mean=2)
    ax.plot(tms_mean, label="TMS (mean)", color="C2", linewidth=3.0)
    ax.plot(tms_median, label="TMS (median)", color="C4", linewidth=3.0)

    for col in tms_cols:
        line = ax.plot(rdata.tms[col], color="C1", linewidth=1.0, alpha=0.5)
    line[0].set(label="TMS")
    
    ax.set(ylabel="Temperature (°C)")
    ax.set_ylim([20, 100])
    
    # ax.legend()  # If you want a legend to be shown (over the graph, makes it unclear)

    i += 1


# +
"""3 RL in"""

fig, axes = plt.subplots(7, 1, figsize=[8, 11])  # the same number of plots as you have groups of runs (all_run_datasets)

i = 0
for runs_str, rdata in all_run_datasets.items():
    #rdata.tms.info()
    
    ax = axes[i]
    #rdata.smart_track.plot(ax=ax)
    ax.plot(rdata.smart_track["RL"], linewidth=3.0, label="Smart Track")
    
    tms_cols = [c for c in rdata.tms.columns if c.startswith("RL_in")]
    
    tms_mean = rdata.get_tms_channels(corner="RL", inout="in", mean=1)
    tms_median = rdata.get_tms_channels(corner="RL", inout="in", mean=2)
    ax.plot(tms_mean, label="TMS (mean)", color="C2", linewidth=3.0)
    ax.plot(tms_median, label="TMS (median)", color="C4", linewidth=3.0)

    for col in tms_cols:
        line = ax.plot(rdata.tms[col], color="C1", linewidth=1.0, alpha=0.5)
    line[0].set(label="TMS")
    
    ax.set(ylabel="Temperature (°C)")
    ax.set_ylim([20, 100])
    
    # ax.legend()  # If you want a legend to be shown (over the graph, makes it unclear)

    i += 1

# +
"""4 RL out"""

fig, axes = plt.subplots(7, 1, figsize=[8, 11])  # the same number of plots as you have groups of runs (all_run_datasets)

i = 0
for runs_str, rdata in all_run_datasets.items():
    #rdata.tms.info()
    
    ax = axes[i]
    #rdata.smart_track.plot(ax=ax)
    ax.plot(rdata.smart_track["RL"], linewidth=3.0, label="Smart Track")
    
    tms_cols = [c for c in rdata.tms.columns if c.startswith("RL_out")]
    
    tms_mean = rdata.get_tms_channels(corner="RL", inout="out", mean=1)
    tms_median = rdata.get_tms_channels(corner="RL", inout="out", mean=2)
    ax.plot(tms_mean, label="TMS (mean)", color="C2", linewidth=3.0)
    ax.plot(tms_median, label="TMS (median)", color="C4", linewidth=3.0)

    for col in tms_cols:
        line = ax.plot(rdata.tms[col], color="C1", linewidth=1.0, alpha=0.5)
    line[0].set(label="TMS")
    
    ax.set(ylabel="Temperature (°C)")
    ax.set_ylim([20, 100])
    
    # ax.legend()  # If you want a legend to be shown (over the graph, makes it unclear)

    i += 1

# +
"""5 FR in"""

fig, axes = plt.subplots(7, 1, figsize=[8, 11])  # the same number of plots as you have groups of runs (all_run_datasets)

i = 0
for runs_str, rdata in all_run_datasets.items():
    #rdata.tms.info()
    
    ax = axes[i]
    #rdata.smart_track.plot(ax=ax)
    ax.plot(rdata.smart_track["FR"], linewidth=3.0, label="Smart Track")
    
    tms_cols = [c for c in rdata.tms.columns if c.startswith("FR_in")]
    
    tms_mean = rdata.get_tms_channels(corner="FR", inout="in", mean=1)
    tms_median = rdata.get_tms_channels(corner="FR", inout="in", mean=2)
    ax.plot(tms_mean, label="TMS (mean)", color="C2", linewidth=3.0)
    ax.plot(tms_median, label="TMS (median)", color="C4", linewidth=3.0)

    for col in tms_cols:
        line = ax.plot(rdata.tms[col], color="C1", linewidth=1.0, alpha=0.5)
    line[0].set(label="TMS")
    
    ax.set(ylabel="Temperature (°C)")
    ax.set_ylim([20, 100])
    
    # ax.legend()  # If you want a legend to be shown (over the graph, makes it unclear)

    i += 1

# +
"""6 FR out"""

fig, axes = plt.subplots(7, 1, figsize=[8, 11])  # the same number of plots as you have groups of runs (all_run_datasets)

i = 0
for runs_str, rdata in all_run_datasets.items():
    #rdata.tms.info()
    
    ax = axes[i]
    #rdata.smart_track.plot(ax=ax)
    ax.plot(rdata.smart_track["FR"], linewidth=3.0, label="Smart Track")
    
    tms_cols = [c for c in rdata.tms.columns if c.startswith("FR_out")]
    
    tms_mean = rdata.get_tms_channels(corner="FR", inout="out", mean=1)
    tms_median = rdata.get_tms_channels(corner="FR", inout="out", mean=2)
    ax.plot(tms_mean, label="TMS (mean)", color="C2", linewidth=3.0)
    ax.plot(tms_median, label="TMS (median)", color="C4", linewidth=3.0)

    for col in tms_cols:
        line = ax.plot(rdata.tms[col], color="C1", linewidth=1.0, alpha=0.5)
    line[0].set(label="TMS")
    
    ax.set(ylabel="Temperature (°C)")
    ax.set_ylim([20, 100])
    
    # ax.legend()  # If you want a legend to be shown (over the graph, makes it unclear)

    i += 1

# +
"""7 FL in"""

fig, axes = plt.subplots(7, 1, figsize=[8, 11])  # the same number of plots as you have groups of runs (all_run_datasets)

i = 0
for runs_str, rdata in all_run_datasets.items():
    #rdata.tms.info()
    
    ax = axes[i]
    #rdata.smart_track.plot(ax=ax)
    ax.plot(rdata.smart_track["FL"], linewidth=3.0, label="Smart Track")
    
    tms_cols = [c for c in rdata.tms.columns if c.startswith("FL_in")]
    
    tms_mean = rdata.get_tms_channels(corner="FL", inout="in", mean=1)
    tms_median = rdata.get_tms_channels(corner="FL", inout="in", mean=2)
    ax.plot(tms_mean, label="TMS (mean)", color="C2", linewidth=3.0)
    ax.plot(tms_median, label="TMS (median)", color="C4", linewidth=3.0)

    for col in tms_cols:
        line = ax.plot(rdata.tms[col], color="C1", linewidth=1.0, alpha=0.5)
    line[0].set(label="TMS")
    
    ax.set(ylabel="Temperature (°C)")
    ax.set_ylim([20, 100])
    
    # ax.legend()  # If you want a legend to be shown (over the graph, makes it unclear)

    i += 1

# +
"""8 FL out"""

fig, axes = plt.subplots(7, 1, figsize=[8, 11])  # the same number of plots as you have groups of runs (all_run_datasets)

i = 0
for runs_str, rdata in all_run_datasets.items():
    #rdata.tms.info()
    
    ax = axes[i]
    #rdata.smart_track.plot(ax=ax)
    ax.plot(rdata.smart_track["FL"], linewidth=3.0, label="Smart Track")
    
    tms_cols = [c for c in rdata.tms.columns if c.startswith("FL_out")]
    
    tms_mean = rdata.get_tms_channels(corner="FL", inout="out", mean=1)
    tms_median = rdata.get_tms_channels(corner="FL", inout="out", mean=2)
    ax.plot(tms_mean, label="TMS (mean)", color="C2", linewidth=3.0)
    ax.plot(tms_median, label="TMS (median)", color="C4", linewidth=3.0)

    for col in tms_cols:
        line = ax.plot(rdata.tms[col], color="C1", linewidth=1.0, alpha=0.5)
    line[0].set(label="TMS")
    
    ax.set(ylabel="Temperature (°C)")
    ax.set_ylim([20, 100])
    
    # ax.legend()  # If you want a legend to be shown (over the graph, makes it unclear)

    i += 1
# -


