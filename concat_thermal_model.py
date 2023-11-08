# +
import numpy as np

import pandas

from matplotlib import pyplot as plt

from nptdms import TdmsFile

from pathlib import Path

from scipy import signal

from scipy.interpolate import interp1d

from scipy.optimize import minimize

from scipy.integrate import solve_ivp

from dataclasses import dataclass

from datetime import datetime, timedelta

from pyro.dynamic import statespace

import pyro

import re

import pickle

plt.style.use("ggplot")

# %matplotlib widget

# +
def grp2df(group, channels):
    channels["Time"] = "Time"
    series = {k: group[v][:] for k, v in channels.items()}
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
    ) - timedelta(seconds=int(frow.name))
    end_dt = start_dt + timedelta(seconds=int(lrow.name))
    return (start_dt, end_dt)

def set_tdindex(df, start: datetime):
    td = (df.index.values * 1000).astype("timedelta64[ms]")
    start_td64 = np.datetime64(int(start.timestamp()*1000), 'ms')
    tdidx = start_td64 + td
    return df.set_index(tdidx)

@dataclass
class RunData:
    probe: pandas.DataFrame
    smart_track: pandas.DataFrame
    malt: pandas.DataFrame
    can: pandas.DataFrame
    gps: pandas.DataFrame
    ir: pandas.DataFrame

def load_run(run: int):
    try:
        this_file_dir = Path(__file__).parent.resolve()
    except NameError:
        this_file_dir = Path(".").resolve()

    datadir = Path(r"\\meca.gme.usherbrooke.ca\CREATEK\Contrats\Michelin S4S\Contrat 2\Michelin test data")
    datafile = next(datadir.glob(f"DT-3105-RUN{run}_*.tdms"))
    with TdmsFile.open(datafile) as f:
        for grp in f.groups():
            print(grp.name)
            print(grp.properties)
            for chan in grp.channels():
                unit = chan.properties["unit_string"]
                #print(f"  {chan.name:25} {unit}")


        gps_channels = {
            "year": f"GPS@year.RN_{run}",
            "month": f"GPS@month.RN_{run}",
            "day": f"GPS@day.RN_{run}",
            "hour": f"GPS@hour.RN_{run}",
            "minute": f"GPS@minute.RN_{run}",
            "second": f"GPS@second.RN_{run}",
            "nanosecond": f"GPS@nsec.RN_{run}",
            "nsat": f"GPS@nsat.RN_{run}",
            "speed_ms": f"GPS@speed_ms.RN_{run}"
        }
        gps_data = grp2df(f["Group1"], gps_channels)

        probe_channels = [c.name for c in f["Group5"].channels()]
        probe_chan_names = [clean_probe_name(s) for s in probe_channels]
        probe_data = grp2df(f["Group5"], dict(zip(probe_chan_names, probe_channels)))

        smart_track_channels = [c.name for c in f["Group4"].channels() if c.name.startswith("ISO@Temp_")]
        smart_track_chan_names = [clean_smarttrack_name(s) for s in smart_track_channels]
        smart_track_data = grp2df(f["Group4"], dict(zip(smart_track_chan_names, smart_track_channels)))

        # Mid Axle Load Transducers
        malt_channels = [c.name for c in f["Group4"].channels() if c.name.startswith("Load@Calc")]
        malt_chan_names = [clean_load_name(s) for s in malt_channels]
        malt_data = grp2df(f["Group4"], dict(zip(malt_chan_names, malt_channels)))

        can_navspeed_chan = [c for c in f["Group4"] if c.startswith("ISO@NavSpeed")][0]
        can_data = grp2df(
            f["Group4"],
            {"NavSpeed": can_navspeed_chan}
        )
        
        # IR and straddle
        ir_channels = {clean_IR_name(c.name): c.name for c in f["Group3"].channels() if c.name.startswith("IR")}
        ir_channels["straddle"] = f"Straddle@Straddle.RN_{run}"
        all_ir_data = grp2df(f["Group3"], ir_channels)
        all_ir_data["straddle"] = all_ir_data["straddle"].astype(bool)
        
    start, end = get_start_end_time(gps_data)
    if start is not None:
        probe_data = set_tdindex(probe_data, start)
        smart_track_data = set_tdindex(smart_track_data, start)
        malt_data = set_tdindex(malt_data, start)
        can_data = set_tdindex(can_data, start)
        all_ir_data = set_tdindex(all_ir_data, start)
        
    res = RunData(
        probe=probe_data,
        smart_track=smart_track_data,
        malt=malt_data,
        can=can_data,
        gps=gps_data,
        ir=all_ir_data
    )
    return res


# +
def get_dt(df):
    """Extract seconds time array from DF with datetime index"""
    return (df.index.values - df.index.values[0]) / np.timedelta64(1, 's')

def interp_at(series, new_index):
    data = np.interp(new_index, series.index, series)
    return pandas.Series(data=data, index=new_index, name=series.name)

def lpfilt_df(df, fc):
    """Low-pass filter all columns of a dataframe with datetime index"""
    res = pandas.DataFrame(index=df.index.copy())
    dt = get_dt(df)
    fs_hz = 1 / (dt[2] - dt[1])
    print(f"{fs_hz=}")
    lpfilt = signal.iirfilter(2, Wn=fc, btype="lowpass", ftype="bessel", output="sos", fs=fs_hz)

    for colname, col in df.items():
        res[colname] = signal.sosfiltfilt(lpfilt, col)
    
    return res

def despike_df(df, wsize, rt, at=0):
    res = pandas.DataFrame(index=df.index.copy())

    for colname, col in df.items():
        rolling=col.rolling(wsize, center=True)
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
    dt = get_dt(df)
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

    # Keep only data when tractor is stopped
    speed_interp_probe_idx = interp_at(candata["NavSpeed"], df.index)
    is_stopped = abs(speed_interp_probe_idx) < 0.2
    #probe_clean[~is_stopped] = np.nan
    return probe_clean

def probe_state_machine(probe_filt_df, speed_kmh, startcond="in"):
    state = pandas.Series(index=probe_filt_df.columns, data=startcond, dtype=object)
    speed_interp = interp_at(speed_kmh, probe_filt_df.index)
    der = df_derivative(probe_filt_df)
    res = probe_filt_df.copy()
    for time, row in probe_filt_df.iterrows():
        #print(state)
        for colname, val in row.items():
            curstate = state.loc[colname]
            # Transitions
            if curstate == "in":
                if abs(speed_interp.loc[time]) < 0.2 and der.loc[time, colname] < -1.0:
                    state.loc[colname] = "tran_out"
                    
            elif curstate == "tran_out":
                if abs(speed_interp.loc[time]) < 0.2 and der.loc[time, colname] > -1.0:
                    state.loc[colname] = "out"

                    
            elif curstate == "out":
                if abs(speed_interp.loc[time]) < 0.2 and der.loc[time, colname] > 1.0:
                    state.loc[colname] = "tran_in"
            
            elif curstate == "tran_in":
                if abs(speed_interp.loc[time]) < 0.2 and der.loc[time, colname] < 1.0:
                    state.loc[colname] = "in"
                    
            # Actions
            if state.loc[colname] != "in":
                res.loc[time, colname] = np.nan
    return res
                
def clean_smart_track(df):
    r = df.copy()
    r[r > 200] = np.nan
    r[r < 0] = np.nan
    return r

def clean_data(d: RunData, probe_init) -> RunData:
    result = RunData(None, None, None, None, None, None)
    probe_clean = clean_probe_data(d.probe, d.can)
    result.probe = probe_state_machine(probe_clean, d.can["NavSpeed"], probe_init)
       
    result.smart_track = clean_smart_track(d.smart_track)
    
    result.can = d.can
    result.can["NavSpeed"][result.can["NavSpeed"] > 80.0] = 0.0
    result.gps = d.gps
    result.malt = lpfilt_df(d.malt, fc=0.01)
    result.ir = d.ir
    return result


# -

data5 = clean_data(load_run(5), "in")
data6 = clean_data(load_run(6), "out")
data7 = clean_data(load_run(7), "out")

print(data5.ir.info())
data5.ir.head()


def concat_data(objs):
    result = RunData(None, None, None, None, None, None)
    result.probe = pandas.concat([o.probe for o in objs], axis=0, verify_integrity=True)
    result.can = pandas.concat([o.can for o in objs], axis=0, verify_integrity=True)
    result.malt = pandas.concat([o.malt for o in objs], axis=0, verify_integrity=True)
    result.smart_track = pandas.concat([o.smart_track for o in objs], axis=0, verify_integrity=True)
    result.ir = pandas.concat([o.ir for o in objs], axis=0, verify_integrity=True)
    return result


# +
cdata = concat_data([data5, data6, data7])
ax = cdata.probe.plot(figsize=[9, 7])
ax.set(ylabel="Temperature (°C)")

ax = cdata.can.plot(figsize=[9, 7])
ax.set(ylabel="Speed (km/h)")
ax.get_legend().remove()

ax = cdata.malt.plot(figsize=[9, 7])
ax.set(ylabel="Load (N)")

ax = cdata.smart_track.plot(figsize=[9, 7])
ax.set(ylabel="Temperature (°C)")

fig, ax = plt.subplots(1, 1, figsize=[9, 7])
ax.plot(cdata.ir.index, cdata.ir.straddle)

with open("concat_runs_5_6_7.pickle", "wb") as f:
    pickle.dump(cdata, f)
# -

# ## Prepare data used for thermal model fit

# +
smart_track_fr_columns = [c for c in cdata.smart_track.columns if c[:2] in "FR"]
fig, ax = plt.subplots()
cdata.smart_track[smart_track_fr_columns].mean(axis=1).dropna().rolling(600).mean()[::100].plot(marker=".", ls="none")

#smart_track_fr = cdata.smart_track[smart_track_fr_columns].mean(axis=1).set_axis(get_dt(cdata.smart_track)).dropna()

# +
class TwoThermalMass( statespace.StateSpaceSystem ):
    """

    """

    ############################
    def __init__(self):
        """ """

        # Initial guesses for parameters
        self.c1  =  1000 #[J/K]
        self.c2  =  1000 #[J/K]
        self.r10 =  3    #[K/watts]
        self.r12 =  6    #[K/watts]
        self.r20 =  3    #[K/watts]

        
        # Matrix ABCD
        self.compute_ABCD()
        
        # initialize standard params
        statespace.StateSpaceSystem.__init__( self, self.A, self.B, self.C, self.D)
        
        # Name and labels
        self.name = 'TwoThermalMass'
        self.input_label = [ 'Heat generation (Qdot)', 'Ambient Temp']
        self.input_units = [ '[W]', '[°C]']
        self.output_label = ['Hotspot Temp.','Smart track Temp.']
        self.output_units = [ '[°C]', '[°C]']
        self.state_label = [ 'Hotspot Temp.','Smart track Temp.']
        self.state_units = [ '[°C]', '[°C]']
        
    
    ###########################################################################
    def compute_ABCD(self):
        """ 
        """
        c1 = self.c1
        c2 = self.c2
        r12 = self.r12
        r10 = self.r10
        r20 = self.r20
        
        
        self.A = np.array([ [ -1/c1*(1./r10+1./r12) ,  +1/(c1 * r12)          ], 
                            [ +1/(c1 * r12)         ,  -1/c2*(1./r20+1./r12)] ])
        
        self.B =  np.array([ [ 1/c1  ,  1/(c1 * r10)   ], 
                             [ 0     ,  1/(c2 * r20)   ] ])
        
        self.C = np.array([[ 0. , 1. ]]) # Note: single output = T2, track sensor temp
        
        self.D = np.array([[ 0. , 0. ]])

sys = TwoThermalMass()


# +
def get_dt(df):
    """Extract seconds time array from DF with datetime index"""
    return (df.index.values - df.index.values[0]) / np.timedelta64(1, 's')


interp_speed = interp1d(get_dt(cdata.can), cdata.can.NavSpeed.values, fill_value=0, bounds_error=False)
interp_load_frin = interp1d(get_dt(cdata.malt), cdata.malt.FR_in.values, fill_value=0, bounds_error=False)

# +
ambient_temp = 22 # deg C

def get_t2u_fun(Kh):
    def t2u(t):
        load = np.array(interp_load_frin(t), ndmin=1)
        load[load < 0.0] = 0.0
        speed = interp_speed(t)
        Qdot_in = np.asarray(Kh * np.abs(load)**2.0 * speed)
        if Qdot_in.ndim == 0:
            Qdot_in = Qdot_in.reshape([1,])
        Ta_arr = np.full_like(Qdot_in, ambient_temp)
        result= np.stack([Qdot_in, Ta_arr], axis=1)
        assert np.all(np.isfinite(result))
        if result.shape[0] == 1: result = result.reshape([2,])
        return result
    return t2u


# -

# ## Determine initial conditions
#
# Hotspot temp and smart track temp

# +
fig, axes = plt.subplots(1, 2, figsize=[12, 4])
axes[0].plot(cdata.probe.probe_FR_IN.iloc[:100])
x0_hs = cdata.probe.probe_FR_IN.iloc[:100].mean()

# No FR data for smart track at beginning of test, so we average all smart
# track channels as an estimate of initial conditions.
axes[1].plot(cdata.smart_track.iloc[:100, :])
x0_st = cdata.smart_track.iloc[:100, :].mean().mean()

x0 = np.array([x0_hs, x0_st])
print(x0)


# -

# ## Optimize system parameters (system identification)
#
# Parameters to optimize:
#
#   * Thermal resistances $r_{10}$, $r_{12}$, $r_{20}$
#   * Thermal masses $c_1$, $c_2$
#   * Heating factor $K_h$ 
#   
# $$ \dot{Q}_{in} = K_h \cdot \left[ \epsilon \left( F_a \right ) \right]^2 \cdot V $$

# +
def eval_cost(traj, t1_df, t2_df):
    interp_t1 = interp1d(traj.t, traj.x[:, 0])(t1_df.index.values)
    interp_t2 = interp1d(traj.t, traj.x[:, 1])(t2_df.index.values)
    num_pts = interp_t1.shape[0] + interp_t2.shape[0]
    cost_t1 = ((t1_df.values - interp_t1) ** 2).sum()
    cost_t2 = ((t2_df.values - interp_t2) ** 2).sum()
    return (cost_t1 + cost_t2) / num_pts

def create_sys_constQin(optim_vars):
    """
    Linear TwoThermalMass model with Qin = 1E-7 * V * F**2
    """
    sys = TwoThermalMass()
    optim_vars = np.asarray(optim_vars)
    sys.c1 = optim_vars[0] * 1000
    sys.c2 = optim_vars[1] * 1000
    sys.r10 = optim_vars[2]
    sys.r12 = optim_vars[3]
    sys.r20 = optim_vars[4]
    sys.compute_ABCD()
    sys.t2u = get_t2u_fun(1E-7)
    sys.x0  = np.array([102, 45])
    return sys



# +
smart_track_fr_columns = [c for c in cdata.smart_track.columns if c.startswith("FR")]
#cdata.smart_track[smart_track_fr_columns].plot()

smart_track_fr = cdata.smart_track[smart_track_fr_columns].mean(axis=1).set_axis(get_dt(cdata.smart_track)).dropna()
smart_track_fr = smart_track_fr.rolling(400).mean().dropna()
smart_track_fr = smart_track_fr[::360]

probe_fr_in = cdata.probe.probe_FR_IN.set_axis(get_dt(cdata.probe)).dropna()
probe_fr_in = probe_fr_in[::50]

probe_fr_in = pandas.concat([
    probe_fr_in[probe_fr_in.index < 4000][::10],
    probe_fr_in[(probe_fr_in.index > 4000) & (probe_fr_in.index < 8000)],
    probe_fr_in[(probe_fr_in.index > 8000)][::3]
]
)
print(smart_track_fr.shape)
print(probe_fr_in.shape)

fig, ax = plt.subplots()
ax.plot(smart_track_fr, ".", label="Smart track FR")
ax.plot(probe_fr_in, ".", label="Probe FR_in")
ax.legend()

assert(np.all(np.isfinite(probe_fr_in)))
assert(np.all(np.isfinite(smart_track_fr)))


# +
plt.close("all")
xg = [1, 1, 25, 5, 3.5] 
sys = create_sys_constQin(xg)
traj = sys.compute_trajectory(tf=9540, n=2000, rtol=1E-6)

fig, axes = plt.subplots(2, 1, figsize=[9, 9])
axes[0].plot(smart_track_fr, ".", label="Smart track FR")
axes[0].plot(probe_fr_in, ".", label="Probe FR_in")
axes[0].plot(traj.t, traj.x[:, 0], label="Simulation hotspot")
axes[0].plot(traj.t, traj.x[:, 1], label="Simulation sensor")
axes[0].set(ylabel="Temperature (°C)")
axes[0].legend()

axes[1].plot(traj.t, traj.u[:, 0])
axes[1].set(ylabel="Heat input (W)")

cost = eval_cost(traj, probe_fr_in, smart_track_fr)
print(f"{cost=:.1e}")


# -

def optimize_model(func_create_sys, x0, t1_df, t2_df):
    """Optimize model parameters to best fit experimental data.
    
    Parameters
    -----------------
    
    func_create_sys: Callable
                     Function with signature `f(param_array): sys`. Takes as an input the parameters
                     to be optimized (N x 1 array-like) and returns an instance of a pyro
                     `ContinuousDynamicSystem`.
    
    x0:              Nx1 array-like
                     Initial guess for the model parameters
    
    t1_df, t2_df:    Pandas series of hotspot temperature data (t1) and measurable temperature
                     data (t2), with a consistent seconds index.
                     
    Returns
    --------------
    
    Result of `scipy.optimize.minimize`
    
    """
    
    # Time for model simulation
    tf_sim = max([t1_df.index.values.max(), t2_df.index.values.max()]) + 1
    
    def optim_cost_fun(params):
        sys = func_create_sys(params)
        traj = sys.compute_trajectory(tf=tf_sim, n=2000, rtol=1E-6)
        cost = eval_cost(traj, t1_df, t2_df)
        return cost

    def cb(xk):
        cost = optim_cost_fun(xk)
        print(xk, " ", cost)

    optimres = minimize(
        optim_cost_fun,
        x0=x0,
        method="Powell",
        tol=0.1,
        callback=cb
    )
    return optimres


# +
def set_axis_dt(ser):
    return ser.set_axis(get_dt(ser))

V_df = set_axis_dt(cdata.can.NavSpeed)
fa_df = set_axis_dt(cdata.malt.FR_in)

optimres_constQin = optimize_model(
    create_sys_constQin,
    x0=[1, 1, 25, 5, 3.5],
    t1_df=probe_fr_in,
    t2_df=smart_track_fr
)
print(optimres_constQin)
print(optimres_constQin.x)

# +
plt.close("all")

sys = create_sys_constQin(optimres_constQin.x)
print("A=")
print(sys.A)
print("")

# Print eigenvalues of A
(eigvals, eigvecs) = np.linalg.eig(sys.A)
print("Eigenvalues of A:")
print(eigvals)
print("")

print("B=")
print(sys.B)
print("")

print("C=")
print(sys.C)
print("")

print("D=")
print(sys.D)
print("")

traj = sys.compute_trajectory(tf=9540, n=2000, rtol=1E-6)

fig, axes = plt.subplots(2, 1, figsize=[11, 9])

axes[0].plot(smart_track_fr, ".", label="Smart track FR")
axes[0].plot(probe_fr_in, ".", label="Probe FR_in")
axes[0].plot(traj.t, traj.x[:, 0], label="Simulation hotspot")
axes[0].plot(traj.t, traj.x[:, 1], label="Simulation sensor (smart track)")
axes[0].set(xlabel="Time (s)", ylabel="Temperature (°C)")
axes[0].legend()

ta = axes[1].twinx()
axes[1].plot(V_df, label="Tractor speed")
axes[1].set(ylim=[-1, 35], ylabel="Speed (km/h)")
axes[1].legend()

ta.plot(fa_df, color="C1", label="Axle load")
ta.set(ylim=[-200, 9000], ylabel="Load (lbf)")
ta.grid(False)
ta.legend()

fig, ax = plt.subplots(figsize=[11, 4])
ax.plot(traj.t, traj.u)


# -
# ## Temperature-dependent $tan(\delta)$
#
# Here we attempt fitting the same thermal model, but we use the following model for the heat generation $\dot{Q}_{in}$.
#
# $$ \dot{Q}_{in} = \tan(\delta) \cdot F_a^2 \cdot V $$
#
# where
#
# $$ \tan(\delta) = m \cdot T_1 + b $$
#
# is the temperature-dependent loss modulus of the rubber. We will therefore be adding $m$ and $b$ as parameters to be fit to the model.

# +
class TwoThermalMassTempt2u(TwoThermalMass):
    """
    Same as `TwoThermalMass` but t2u is state-dependent with signature
    `u = t2u(x, t)`.
    """
    def __init__(self, c1, c2, r10, r12, r20, m_delta, b_delta):
        super().__init__()
        
        self.c1 = c1
        self.c2 = c2
        self.r10 = r10
        self.r12 = r12
        self.r20 = r20
        super().compute_ABCD()
        
        self.m_delta = m_delta
        self.b_delta = b_delta
        
    def f(self, x, u, t):
        # Modify input u to add temperature-dependent input
        t1 = x[0]
        Qdot_in_const = u[0]
        tan_delta = np.asarray(self.m_delta * t1 + self.b_delta)
        tan_delta[tan_delta < 0] = 0.0
        
        Qdot_in = tan_delta * Qdot_in_const
        
        u_mod = np.array([Qdot_in, u[1]])
        return super().f(x, u_mod, t)
        
def create_sys_tempt2u(optim_vars):
    """
    Linear TwoThermalMass model with Qin = 1E-7 * tan(delta) * V * F**2
    where tan(delta) = m * T1 + b
    """
    optim_vars = np.asarray(optim_vars)
    
    sys = TwoThermalMassTempt2u(
        c1 = optim_vars[0] * 1000,
        c2 = optim_vars[1] * 1000,
        r10 = optim_vars[2],
        r12 = optim_vars[3],
        r20 = optim_vars[4],
        m_delta = optim_vars[5] * 0.01,
        b_delta = optim_vars[6],
    )
    
    sys.t2u = get_t2u_fun(1E-7)
    sys.x0  = np.array([102, 50])
    return sys


# +
xg = [1.818, 1.782, 5, 4.677, 3.421, -3.5, 5]
sys = create_sys_tempt2u(xg)
traj = sys.compute_trajectory(tf=9540, n=2000, rtol=1E-6)

fig, ax = plt.subplots()
ax.plot(smart_track_fr, ".", label="Smart track FR")
ax.plot(probe_fr_in, ".", label="Probe FR_in")
ax.plot(traj.t, traj.x[:, 0], label="Simulation hotspot")
ax.plot(traj.t, traj.x[:, 1], label="Simulation sensor (smart track)")
ax.set(xlabel="Time (s)", ylabel="Temperature (°C)")
ax.legend()

fig, ax = plt.subplots()
ax.plot(traj.t, traj.u)
# -

res = optimize_model(
    create_sys_tempt2u,
    x0=[1.818, 1.782, 5, 4.677, 3.421, -3.5, 5],
    t1_df=probe_fr_in,
    t2_df=smart_track_fr
)
print(res)


def mask_straddle(load_data, rd: RunData):
    straddle_bool = set_axis_dt(rd.ir.straddle)
    straddle_fa = pandas.DataFrame(load_data.copy())
    straddle_fa = pandas.merge_asof(straddle_fa, straddle_bool, left_index=True, right_index=True)
    straddle_fa.loc[~straddle_fa["straddle"], "FR_in"] = np.nan
    return straddle_fa


# +
sys = create_sys_tempt2u(res.x)

traj = sys.compute_trajectory(tf=9540, n=2000, rtol=1E-6)

fig, axes = plt.subplots(2, 1, figsize=[11, 9])

axes[0].plot(smart_track_fr, ".", label="Smart track FR")
axes[0].plot(probe_fr_in, ".", label="Probe FR_in")
axes[0].plot(traj.t, traj.x[:, 0], label="Simulation hotspot")
axes[0].plot(traj.t, traj.x[:, 1], label="Simulation sensor (smart track)")
axes[0].set(xlabel="Time (s)", ylabel="Temperature (°C)")
axes[0].legend()

ta = axes[1].twinx()
axes[1].plot(V_df, label="Tractor speed")
axes[1].set(ylim=[-1, 35], ylabel="Speed (km/h)")
axes[1].legend()

fa_straddle = mask_straddle(fa_df, cdata)
ta.plot(fa_straddle["FR_in"], color="C4", lw=6, label="Straddle")

ta.plot(fa_df, color="C1", label="Axle load")
ta.set(ylim=[-200, 9000], ylabel="Load (lbf)")
ta.grid(False)
ta.legend()

fig, ax = plt.subplots(figsize=[11, 4])
ax.plot(traj.t, traj.u)

# +
m_fit = sys.m_delta
b_fit = sys.b_delta

tt = np.linspace(15, 160, 20)
tan_delta_eval = m_fit * tt + b_fit
fig, ax = plt.subplots(figsize=[9, 5])
ax.plot(tt, tan_delta_eval)
ax.set(ylabel="$tan(\delta)$", xlabel="$T_1$ (°C)")

print(res.x)
# -


