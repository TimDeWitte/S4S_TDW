import numpy as np

import pandas

from nptdms import TdmsFile

from pathlib import Path

from scipy import signal

from scipy.interpolate import interp1d

from dataclasses import dataclass

import dataclasses

from datetime import datetime, timedelta, UTC

import re

import pickle


def grp2df(group, channels):
    series_temp = {k: group[v][:] for k, v in channels.items()}
    shortest_series = min([c.shape[0] for c in series_temp.values()])
    channels["Time"] = "Time"
    series = {k: group[v][:shortest_series] for k, v in channels.items()}

    df = pandas.DataFrame(data=series)
    df = df.set_index("Time")
    return df

def clean_probe_name(name: str) -> str:
    pat = re.compile(r"Probes@(\w+)(\..*)?")
    m = pat.match(name)
    if m is not None:
        return m.group(1)
    else:
        return name


def clean_smarttrack_name(name: str) -> str:
    pat = re.compile(r"ISO@Temp_(A\d)_(S\d)(\..*)?")
    m = pat.match(name)
    lut = {"A1": "FR", "A2": "FL", "A3": "RR", "A4": "RL"}
    if m is not None:
        return f"{(lut[m.group(1)])}_{m.group(2)}"
    else:
        return name


def clean_load_name(name: str) -> str:
    pat = re.compile(r"Load@Calc_(\w+)(\..*)?")
    m = pat.match(name)
    if m is not None:
        return m.group(1)
    else:
        return name


def clean_IR_name(name: str) -> str:
    result = name.split("@")[1]
    result = result.split(".")[0]
    return result


def get_start_end_time(gps_data):
    rows_filtered = gps_data[gps_data.year == 2022.0]
    if len(rows_filtered) == 0:
        return (None, None)
    frow = rows_filtered.iloc[0]
    lrow = gps_data.iloc[-1]
    start_dt = datetime(
        year=int(frow.year),
        month=int(frow.month),
        day=int(frow.day),
        hour=int(frow.hour),
        minute=int(frow.minute),
        second=(int(frow.second) % 60),
        tzinfo=UTC,
    ) - timedelta(seconds=int(frow.name))
    end_dt = start_dt + timedelta(seconds=int(lrow.name))
    return (start_dt, end_dt)


def set_tdindex(df, start: datetime):
    delta = (df.index.values * 1000).astype("timedelta64[ms]")
    start_dt64 = np.datetime64(start)
    tdidx = pandas.to_datetime(start_dt64 + delta, utc=True)
    return df.set_index(tdidx)


@dataclass
class RunData:
    probe: pandas.DataFrame
    probe_state: pandas.DataFrame
    smart_track: pandas.DataFrame
    malt: pandas.DataFrame
    can: pandas.DataFrame
    gps: pandas.DataFrame
    ir: pandas.DataFrame
    tms: pandas.DataFrame

    def get_probe_in(self, corner=None, inout=None):
        res = self.probe.copy()
        res[self.probe_state != "in"] = np.nan
        if corner is not None and inout is not None:
            probe_col = f"probe_{corner.upper()}_{inout.upper()}"
            res = res[probe_col]
        return res

    def get_ambient_est(self, fill=True):
        """Estimate ambient temp based on probes when not inserted"""
        ambt = self.probe.copy()
        ambt[self.probe_state != "out"] = np.nan
        ambt = ambt.min(axis=1)
        med = ambt.rolling(60 * 3, center=True).median()
        if fill:
            med = (
                med.interpolate(method="linear")
                .fillna(method="ffill")
                .fillna(method="bfill")
            )
        return pandas.DataFrame(data={"ambient": med})

    def get_tms_channels(self, corner: str, inout: str, mean=0):  # Update by TDW on 19/10/23 to be able to visualize and work with both mean and median
        prefix = corner.upper() + "_" + inout.lower()
        filtered_columns = [col for col in self.tms.columns if col.startswith(prefix)]
        assert len(filtered_columns) > 0
        data = self.tms[filtered_columns]
        if mean==1:
            # 1. Fill NaNs (missing data when TMS stop transmittig) by forward-fill (last valid data)
            # 2. Average all columns
            # 3. Remove data points where all columns were NaN prior to filling in
            nans_idx = data.isna().all(axis=1)
            data = data.fillna(method="ffill").mean(axis=1)#.mean(axis=1)  # swapped 17/10/23
            data[nans_idx] = np.nan
        if mean==2:
            # 1. Fill NaNs (missing data when TMS stop transmittig) by forward-fill (last valid data)
            # 2. Average all columns
            # 3. Remove data points where all columns were NaN prior to filling in
            nans_idx = data.isna().all(axis=1)
            data = data.fillna(method="ffill").median(axis=1)#.mean(axis=1)  # swapped 17/10/23
            data[nans_idx] = np.nan
        if mean==3:
            # 1. Fill NaNs (missing data when TMS stop transmittig) by forward-fill (last valid data)
            # 2. Average all columns
            # 3. Remove data points where all columns were NaN prior to filling in
            nans_idx = data.isna().all(axis=1)
            data = data.fillna(method="ffill").min(axis=1)#.mean(axis=1)  # swapped 23/10/23
            data[nans_idx] = np.nan
        return data

    def get_malt_channel(self, corner, inout):
        malt_col = f"{corner.upper()}_{inout.lower()}"
        return self.malt[malt_col]

    def get_ir_temp(self, corner=None, inout=None):
        res = self.ir.copy()
        if corner is not None and inout is not None:
            ir_col = f"{corner.upper()}_{inout}"  # f"{corner.upper()}_{inout.upper()}"  # not ir_ !!
            res = res[ir_col]
        return res
    


def get_gps_data(tdms_file: TdmsFile):
    gps_channels_prefix = {
        "year": "GPS@year",
        "month": "GPS@month",
        "day": "GPS@day",
        "hour": "GPS@hour",
        "minute": "GPS@minute",
        "second": "GPS@second",
        "nanosecond": "GPS@nsec",
        "nsat": "GPS@nsat",
        "speed_ms": "GPS@speed_ms",
    }

    gps_channels = {}
    grp = tdms_file["Group1"]
    for chan in grp.channels():
        for k, prefix in gps_channels_prefix.items():
            if chan.name.startswith(prefix):
                gps_channels[k] = chan.name
                break

    gps_data = grp2df(grp, gps_channels)
    return gps_data


def get_tms_data(tdms_file: TdmsFile):
    tms_location_map = {
        "RR_in": [11, 85, 154, 33, 114],
        "RR_out": [116, 8, 81, 64, 14],
        "FR_in": [28, 73, 3, 166],
        "FR_out": [122, 121, 93, 151, 172],
        "RL_in": [51, 153, 104, 68],
        "RL_out": [176, 5, 60, 103],
        "FL_in": [0, 173, 13, 96],  # Was swapped and therefor not correct
        "FL_out": [149, 6, 94, 146, 143, 174],  # Was swapped and therefor not correct
    }

    # Inverse map to ID: location
    tms_id_map = {}
    for loc, id_list in tms_location_map.items():
        for tms_id in id_list:
            tms_id_map[tms_id] = loc

    tms_channels = {}
    grp = tdms_file["Group4"]
    pat_tms_channel = re.compile(r"Wheel_temp_(\d+)")
    for chan in grp.channels():
        m = pat_tms_channel.match(chan.name)
        if m:
            chan_tms_id = int(m.group(1))
            try:
                chan_tms_loc = tms_id_map[chan_tms_id]
            except KeyError:
                continue
            tms_channels[chan_tms_loc + "_" + str(chan_tms_id)] = chan.name

    tms_df = grp2df(grp, tms_channels)
    return tms_df


def get_ir_data(tdms_file: TdmsFile):  # TDW 20/10/23
    ir_location_map = {
        "RR_in": [223],
        "RR_out": [188],
        "FR_in": [224],
        "FR_out": [189],
        "RL_in": [222],
        "RL_out": [221],
        "FL_in": [225],  
        "FL_out": [226],  
        "Straddle": [1000],
        }
    
    # Inverse map to ID: location
    ir_id_map = {}
    for loc, id_list in ir_location_map.items():
        for ir_id in id_list:
            ir_id_map[ir_id] = loc
    
    ir_channels = {}

    grp = tdms_file["Group3"]
    pat_ir_channel_1 = re.compile(r"IR_100_(\d)@IR_(\d+)H(\S)_(\d+)")  # Package 're' for regular expressions to find the IR sensor numbers. 
    # I make from number 22_3 the number 223 and so on
    pat_ir_channel_2 = re.compile(r"IR_100_(\d)@IR_(\d+)H(\S)_(\d+)_(\d)")
    pat_ir_channel_3 = re.compile(r"Straddle@Straddle")
    for chan in grp.channels():
        m = pat_ir_channel_1.match(chan.name)
        n = pat_ir_channel_2.match(chan.name)
        o = pat_ir_channel_3.match(chan.name)
        if m or n or o:  # The required group of characters is found
            if m and not n:
                chan_ir_id = int(m.group(4))
            elif m and n:
                chan_ir_id = int(m.group(4))*10 + int(n.group(5))
            else:
                chan_ir_id = 1000  # The one for the Straddle
            try:
                chan_ir_loc = ir_id_map[chan_ir_id]
            except KeyError:
                continue
            # ir_channels[chan_ir_loc + "_" + str(chan_ir_id)] = chan.name
            ir_channels[chan_ir_loc] = chan.name

    ir_df = grp2df(grp, ir_channels)  # Watch out the Straddle is still in the old file alone (not cleaned)
    return ir_df


def get_dtindex_from_tdms(tdms_file: TdmsFile, reltime_seconds):
    gps_data = get_gps_data(tdms_file)
    start, end = get_start_end_time(gps_data)

    td = (np.asarray(reltime_seconds) * 1000).astype("timedelta64[ms]")
    start_td64 = np.datetime64(int(start.timestamp() * 1000), "ms")
    tdidx = pandas.to_datetime((start_td64 + td), utc=True)
    return tdidx


def load_run(datadir, run: int):
    datafile = next(datadir.glob(f"DT-3105-RUN{run}_*.tdms"))
    with TdmsFile.open(datafile) as f:
        gps_data = get_gps_data(f)
        tms_data = get_tms_data(f)

        probe_channels = [c.name for c in f["Group5"].channels()]
        probe_chan_names = [clean_probe_name(s) for s in probe_channels]
        probe_data = grp2df(f["Group5"], dict(zip(probe_chan_names, probe_channels)))

        smart_track_channels = [
            c.name for c in f["Group4"].channels() if c.name.startswith("ISO@Temp_")
        ]
        smart_track_chan_names = [
            clean_smarttrack_name(s) for s in smart_track_channels
        ]
        smart_track_data = grp2df(
            f["Group4"], dict(zip(smart_track_chan_names, smart_track_channels))
        )

        # Mid Axle Load Transducers
        malt_channels = [
            c.name for c in f["Group4"].channels() if c.name.startswith("Load@Calc")
        ]
        malt_chan_names = [clean_load_name(s) for s in malt_channels]
        malt_data = grp2df(f["Group4"], dict(zip(malt_chan_names, malt_channels)))

        can_navspeed_chan = [c for c in f["Group4"] if c.startswith("ISO@NavSpeed")][0]
        can_grtemp_chan = [c for c in f["Group4"] if c.startswith("IR_7_1@GR_Temp")][0]
        can_data = grp2df(
            f["Group4"], {"NavSpeed": can_navspeed_chan, "GroundTemp": can_grtemp_chan}
        )

        # IR and straddle
        """  # Changed on 20/10/23 because of the written get_ir_data 
        ir_channels = {
            clean_IR_name(c.name): c.name
            for c in f["Group3"].channels()
            if c.name.startswith("IR")
        }
        ir_channels["straddle"] = f"Straddle@Straddle.RN_{run}"
        all_ir_data = grp2df(f["Group3"], ir_channels)
        """
        all_ir_data = get_ir_data(f)  # clean_ir_data to be added here
        all_ir_data["straddle"] = all_ir_data["Straddle"].astype(bool)

    start, end = get_start_end_time(gps_data)
    if start is not None:
        probe_data = set_tdindex(probe_data, start)
        smart_track_data = set_tdindex(smart_track_data, start)
        malt_data = set_tdindex(malt_data, start)
        can_data = set_tdindex(can_data, start)
        all_ir_data = set_tdindex(all_ir_data, start)
        tms_data = set_tdindex(tms_data, start)

    res = RunData(
        probe=probe_data,
        probe_state=None,
        smart_track=smart_track_data,
        malt=malt_data,
        can=can_data,
        gps=gps_data,
        ir=all_ir_data,
        tms=tms_data,
    )
    return res


def get_dt(df, t0):
    """Extract seconds time array from DF with datetime index"""
    if isinstance(t0, pandas.Timestamp):
        t0 = t0.to_numpy()
    if t0 is None:
        t0 = df.index.values[0]
    return (df.index.values - t0) / np.timedelta64(1, "s")


def set_axis_dt(ser, t0):
    return ser.set_axis(get_dt(ser, t0))


def interp_at(series, new_index):
    if np.issubdtype(series.index.values.dtype, np.datetime64):
        assert np.issubdtype(new_index.values.dtype, np.datetime64)
        t0 = new_index.values[0]
        tdelta_new_index = (new_index.values - t0) / np.timedelta64(1, "s")
        tdelta_series = (series.index.values - t0) / np.timedelta64(1, "s")
        data = np.interp(tdelta_new_index, tdelta_series, series.values)
    else:
        data = np.interp(new_index, series.index, series, left=np.nan, right=np.nan)
    return pandas.Series(data=data, index=new_index, name=series.name)


def lpfilt_df(df, fc):
    """Low-pass filter all columns of a dataframe with datetime index"""
    res = pandas.DataFrame(index=df.index.copy())
    dt = get_dt(df, t0=None)
    fs_hz = 1 / (dt[2] - dt[1])
    lpfilt = signal.iirfilter(
        2, Wn=fc, btype="lowpass", ftype="bessel", output="sos", fs=fs_hz
    )

    for colname, col in df.items():
        res[colname] = signal.sosfiltfilt(lpfilt, col)

    return res


def despike_df(df, wsize, rt, at=0):
    res = pandas.DataFrame(index=df.index.copy())

    for colname, col in df.items():
        rolling = col.rolling(wsize, center=True)
        iqr = rolling.quantile(0.75) - rolling.quantile(0.25)
        med = rolling.median()
        ncol = col.copy()
        ncol[abs(ncol - med) > (abs(iqr * rt) + at)] = np.nan
        ncol = ncol.fillna(med)
        res[colname] = ncol

    return res


def df_derivative(df):
    """df must have datetime index"""
    res = pandas.DataFrame(index=df.index.copy())
    dt = get_dt(df, t0=None)
    for colname, col in df.items():
        res[colname] = np.gradient(col, dt, axis=0)
    return res


def clean_probe_data(df, candata):
    probe_clean = df.copy()
    # Remove bad data
    probe_clean[(probe_clean > 200) | (probe_clean < 10)] = np.nan
    probe_clean = probe_clean.fillna(method="ffill")
    probe_clean = despike_df(probe_clean, wsize=5, rt=2.0, at=1.0)

    # Lowpass filter
    probe_clean = lpfilt_df(probe_clean, fc=0.05)
    return probe_clean


def probe_state_machine(probe_filt_df, speed_kmh, startcond="in"):
    state_df = pandas.DataFrame(index=probe_filt_df.index)
    cat_series = pandas.Series(
        pandas.Categorical(
            np.full(len(state_df.index), np.nan),
            categories=["in", "out", "tran_in", "tran_out"],
            ordered=False,
        ),
        index=state_df.index,
    )
    for colname in probe_filt_df.columns:
        state_df[colname] = cat_series.copy()

    state_df.iloc[0, :] = startcond
    prev_row = state_df.iloc[0]
    speed_interp = interp_at(speed_kmh, probe_filt_df.index)
    der = df_derivative(probe_filt_df)
    for time, row in probe_filt_df.iterrows():
        state_df.loc[time] = prev_row
        for colname, val in row.items():
            curstate = prev_row.loc[colname]
            # Transitions
            if curstate == "in":
                if abs(speed_interp.loc[time]) < 0.2 and der.loc[time, colname] < -1.0:
                    state_df.loc[time, colname] = "tran_out"

            elif curstate == "tran_out":
                if abs(speed_interp.loc[time]) < 0.2 and der.loc[time, colname] > -1.0:
                    state_df.loc[time, colname] = "out"

            elif curstate == "out":
                if abs(speed_interp.loc[time]) < 0.2 and der.loc[time, colname] > 1.0:
                    state_df.loc[time, colname] = "tran_in"

            elif curstate == "tran_in":
                if abs(speed_interp.loc[time]) < 0.2 and der.loc[time, colname] < 1.0:
                    state_df.loc[time, colname] = "in"
            else:
                assert False
        prev_row = state_df.loc[time]

    return state_df


def avg_smarttrack_corners(stdf):
    corners = ["FL", "FR", "RL", "RR"]
    corners_df = pandas.DataFrame(index=stdf.index)
    for cnr in corners:
        cols = [c for c in stdf.columns if c.startswith(cnr)]
        corners_df[cnr] = stdf[cols].mean(axis=1)
    return corners_df


def clean_smart_track(df):
    r = df.copy()
    r[r > 200] = np.nan
    r[r < 0] = np.nan
    return avg_smarttrack_corners(r)


def clean_can_data(can_data):
    df = can_data.copy()
    df["NavSpeed"][df["NavSpeed"] > 80.0] = 0.0
    df["GroundTemp"] = lpfilt_df(df[["GroundTemp"]], fc=0.05)["GroundTemp"]
    return df


def detect_linear(ser):
    """Detect data that is perfectly linear.

    Assumes n >= 3 and all points are evenly spaced.
    """
    n = len(ser)

    # Slope and offset
    m = (ser[-1] - ser[0]) / (n - 1)
    b = ser[0]

    linear = m * np.arange(n) + b
    sqres = ((linear - ser) ** 2).sum()
    sstot = ((ser - ser.mean()) ** 2).sum() + 1e-6

    return 1 - sqres / sstot


def clean_tms_data(tms_data):
    # Downsample to period of 10 s
    tms_downsampled_index = pandas.date_range(
        start=tms_data.index[0], end=tms_data.index[-1], freq="10S", unit="us"
    )
    tms_clean = pandas.DataFrame(index=tms_downsampled_index)
    for col in tms_data.columns:
        tms_clean[col] = interp_at(tms_data[col], tms_clean.index)

    # Remove linear interpolated data
    det = tms_clean.rolling(window=6, center=True).apply(detect_linear)
    tms_clean[det > 0.999] = np.nan
    tms_clean[np.isnan(det)] = np.nan

    return tms_clean


def clean_ir_data(ir_data):  
    """To finalize still, but take into account that the last column is the straddle 
    column with booleans True and False and so it does not need to be having the linear """
    # Downsample to period of 10 s
    ir_downsampled_index = pandas.date_range(
        start=ir_data.index[0], end=ir_data.index[-1], freq="10S", unit="us"
    )
    ir_clean = pandas.DataFrame(index=ir_downsampled_index)
    for col in ir_data.columns:
        ir_clean[col] = interp_at(ir_data[col], ir_clean.index)

    # Remove linear interpolated data
    det = ir_clean.rolling(window=6, center=True).apply(detect_linear)
    ir_clean[det > 0.999] = np.nan
    ir_clean[np.isnan(det)] = np.nan

    return ir_clean


def clean_data(d: RunData, probe_init) -> RunData:
    result = RunData(None, None, None, None, None, None, None, None)
    result.probe = clean_probe_data(d.probe, d.can)
    result.probe_state = probe_state_machine(
        result.probe, d.can["NavSpeed"], probe_init
    )

    result.smart_track = clean_smart_track(d.smart_track)

    result.can = clean_can_data(d.can)
    result.gps = d.gps
    result.malt = lpfilt_df(d.malt, fc=0.01)
    result.ir = d.ir  # see the clean_ir_data function (still needs to be finalized)
    result.tms = clean_tms_data(d.tms)

    return result


def concat_data(objs):
    result = RunData(None, None, None, None, None, None, None, None)
    result.probe = pandas.concat([o.probe for o in objs], axis=0, verify_integrity=True)
    result.probe_state = pandas.concat(
        [o.probe_state for o in objs], axis=0, verify_integrity=True
    )
    result.can = pandas.concat([o.can for o in objs], axis=0, verify_integrity=True)
    result.malt = pandas.concat([o.malt for o in objs], axis=0, verify_integrity=True)
    result.smart_track = pandas.concat(
        [o.smart_track for o in objs], axis=0, verify_integrity=True
    )
    result.ir = pandas.concat([o.ir for o in objs], axis=0, verify_integrity=True)
    result.tms = pandas.concat([o.tms for o in objs], axis=0, verify_integrity=True)
    return result


def mask_straddle(df, rd: RunData):
    straddle_bool = set_axis_dt(rd.ir.straddle, t0=None)
    straddle_fa = df.copy()
    straddle_fa = pandas.merge_asof(
        straddle_fa, straddle_bool, left_index=True, right_index=True
    )
    straddle_fa.loc[~straddle_fa["straddle"], :] = np.nan
    return straddle_fa.drop(["straddle"], axis=1)


def crop_data(rd: RunData, start=None, end=None) -> RunData:
    result = rd
    for field in dataclasses.fields(result):
        val = getattr(result, field.name)
        if val is None:
            continue
        if start is not None:
            val = val.loc[val.index >= start]
        if end is not None:
            val = val.loc[val.index <= end]
        setattr(result, field.name, val)
    return result


def get_ambient_temp_weather(t0, src):
    """Estimate ambient temperature based on historical weather data

    Run script / notebook "weather_explore.py" to generate the weather
    data files.
    """
    datadir = Path(__file__).parent

    datasets = {
        "ncei": "ncei_temperatures.pickle",
    }

    if not src in datasets:
        raise ValueError(f'Unknown temperature dataset "{src}"')

    with open((datadir / datasets[src]), "rb") as f:
        temp_data = pickle.load(f)
    dt = get_dt(temp_data, t0)
    return interp1d(
        dt, temp_data.temperature.values, bounds_error=False, fill_value=(22, 22)
    )


def get_ambient_temp_probe(rdata, t0):
    """Estimate ambient temperature based on TC probes"""
    ambient_df = rdata.get_ambient_est()
    print("pre: ", ambient_df)
    ambient_interp = interp1d(
        get_dt(ambient_df, t0),
        ambient_df["ambient"].values,
        fill_value=0,
        bounds_error=False,
    )
    print("post: ", ambient_interp)
    return ambient_interp


def get_ambient_temp_tms(rdata, t0):
    """Estimate ambient temperature based on TMS"""
    ambient_df1 = np.min(rdata.get_tms_channels("FL", "out"), axis=1) # There are multiple TMS channels
    ambient_df = (
                ambient_df1.interpolate(method="linear")
                .fillna(method="ffill")
                .fillna(method="bfill")
            )
    #print("pre: ", ambient_df)
    ambient_interp = interp1d(
            get_dt(ambient_df, t0), ambient_df.values, fill_value="extrapolate" 
        )
    #print("post: ", ambient_interp)
    return ambient_interp


def get_ambient_temp(rdata, t0, src, corner, inout):  # Returns an interpolation function (not a list)!
    if src == "probe":
        return get_ambient_temp_probe(rdata, t0)
    elif src == "ground":
        ambient_df = rdata.can["GroundTemp"]
        #print("pre: ", ambient_df)
        ambient_interp = interp1d(
            get_dt(ambient_df, t0), ambient_df.values, fill_value="extrapolate"
        )
        #print("post: ", ambient_interp)
        return ambient_interp
    elif src == "TMS":  # TDW 1/11/23 added 
        return get_ambient_temp_tms(rdata, t0)
    else:
        return get_ambient_temp_weather(t0, src)




