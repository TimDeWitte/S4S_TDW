# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: camso-s4s-3.11
#     language: python
#     name: camso-s4s-3.11
# ---

# +
import pickle
import pprint

obj = pickle.load(open("./concat_data/runs_29_30_31.pickle", "rb"))
obj2 = pickle.load(open("./concat_data/runs_1_2_3_4.pickle", "rb"))

with open("out.txt", "a") as f:
         pprint.pprint(obj, stream=f)

with open("out2.txt", "a") as f:
         pprint.pprint(obj2, stream=f)
# +
## import pandas
from nptdms import TdmsFile
from pathlib import Path
import pickle
import re

# %matplotlib widget

datadir = Path(
    r"C:\Users\TDewitte\Desktop\PythonPrograms\S4S\camso-s4s\tdms_data"
)

datafile = next(datadir.glob(f"DT-3105-RUN4_Sequence05.tdms"))

ir_location_map = {
        "RR_in": [223],
        "RR_out": [188],
        "FR_in": [224],
        "FR_out": [189],
        "RL_in": [222],
        "RL_out": [221],
        "FL_in": [225],  # Was swapped and therefor not correct
        "FL_out": [226],  # Was swapped and therefor not correct
        "Straddle": [1000],
    }

# Inverse map to ID: location
ir_id_map = {}
for loc, id_list in ir_location_map.items():
    for ir_id in id_list:
        ir_id_map[ir_id] = loc

ir_channels = {}
with TdmsFile.open(datafile) as f:
    all_groups = f.groups()
    print(all_groups)
    group = f["Group3"]
    all_group_channels = group.channels()
    print(all_group_channels)
    pat_ir_channel_1 = re.compile(r"IR_100_(\d)@IR_(\d+)H(\S)_(\d+)")  # re for regular expressions to find the IR sensor numbers. I make from number 22_3 the number 223 and so on
    pat_ir_channel_2 = re.compile(r"IR_100_(\d)@IR_(\d+)H(\S)_(\d+)_(\d)")
    pat_ir_channel_3 = re.compile(r"Straddle@Straddle")
    for chan in group.channels():
        m = pat_ir_channel_1.match(chan.name)
        n = pat_ir_channel_2.match(chan.name)
        o = pat_ir_channel_3.match(chan.name)
        print(o)

        if m or n or o:  # The required group of characters is found
            if m and not n:
                chan_ir_id = int(m.group(4))
            elif m and n: 
                chan_ir_id = int(m.group(4))*10 + int(n.group(5))
            else:
                chan_ir_id = 1000  # The one for the Straddle
            print(chan_ir_id)
            try:
                chan_ir_loc = ir_id_map[chan_ir_id]
                print(chan_ir_loc)
            except KeyError:
                continue
            ir_channels[chan_ir_loc + "_" + str(chan_ir_id)] = chan.name
            print(ir_channels[chan_ir_loc + "_" + str(chan_ir_id)])


# +
import numpy as np
import pandas as pd

data= pd.DataFrame([[1,2,3,4,5,6],[0,5,0,5,0,5]])
minz = data.fillna(method="ffill").min(axis=1)

print(minz)
# +
# %matplotlib tk
import pickle
#figx = pickle.load(open('Residuals_all runs_FR_out.fig.pickle', 'rb'))
figy = pickle.load(open('Model_all runs_RR_out.fig.pickle', 'rb'))

# figx.show()
figy.show()
# -



