# + active=""
# import autoreload
# %load_ext autoreload

# +
# %matplotlib widget

import importlib
imported_module = importlib.import_module("data_utils")
importlib.reload(imported_module)
from data_utils import *


import numpy as np

import pandas

from matplotlib import pyplot as plt

import pickle

from scipy.interpolate import interp1d

import models

import data_utils

import kalman

from pyro.dynamic import statespace

plt.style.use("ggplot")

# #%matplotlib widget

plt.ion()


# +
with open("./concat_data/runs_5_6_7.pickle", "rb") as f:
    rdata_5_6_7 = pickle.load(f)
    
with open("./concat_data/runs_29_30_31.pickle", "rb") as f:
    rdata_29_30_31 = pickle.load(f)
    
with open("./concat_data/runs_18_19_20.pickle", "rb") as f:
    rdata_18_19_20 = pickle.load(f)

# above are the original from Francis

with open("./concat_data/runs_1_2_3_4.pickle", "rb") as f:
    rdata_1_2_3_4 = pickle.load(f)

with open("./concat_data/runs_8_9_10.pickle", "rb") as f:
    rdata_8_9_10 = pickle.load(f)

with open("./concat_data/runs_11_12_13_14.pickle", "rb") as f:
    rdata_11_12_13_14 = pickle.load(f)

with open("./concat_data/runs_15_16_17.pickle", "rb") as f:
    rdata_15_16_17 = pickle.load(f)

with open("./concat_data/runs_18-28.pickle", "rb") as f:
    rdata_18_28 = pickle.load(f)

with open("./concat_data/runs_32_33_34.pickle", "rb") as f:
    rdata_32_34 = pickle.load(f)

with open("./concat_data/runs_all.pickle", "rb") as f:
    rdata_all = pickle.load(f)

    
all_run_datasets = {
    "5, 6, 7": rdata_5_6_7,
    "18, 19, 20": rdata_18_19_20,
    "29, 30, 31": rdata_29_30_31,
    #"1, 2, 3, 4": rdata_1_2_3_4,
    #"8, 9, 10": rdata_8_9_10,
    #"11, 12, 13, 14": rdata_11_12_13_14,
    #"15, 16, 17": rdata_15_16_17,
    #"18-28": rdata_18_28,
    #"32, 33, 34": rdata_32_34
}

all_runs = {"all runs": rdata_all}


# +
def get_t2_data(rdata, src, corner, inout=None):
    if src == "smarttrack":
        t2 = rdata.smart_track[corner.upper()].dropna()
    elif src == "tms":
        if inout is None:
            raise ValueError("inout required for TMS sensors")
        t2 = rdata.get_tms_channels(corner=corner, inout=inout, mean=3).dropna()  # Adjust if it is the mean, median or minimum
    else:
        raise ValueError(f"Invalid t2 src {src}")
    return t2

def dt_cov_penalty(dt):
    """ReLU penalty for old T2 measurement"""
    nominal_dt = 5.0 # seconds, penalize all measurements older than this
    dt = np.abs(np.asarray(dt))
    if dt.ndim == 0:
        dt = dt.reshape(1)
    penalty = 1.0 + (dt - nominal_dt) * 100
    penalty[dt <= nominal_dt] = 1.0
    return penalty

def get_R_fun(R_nom, cdata, T2src, corner, t0, inout=None):
    """Generate function for time-varying sensor noise covariance matrix R(t)
    
    R is set very high for missing sensor data.
    """
    t2 = get_t2_data(cdata, src=T2src, corner=corner, inout=inout)
                     
    st_data_t = data_utils.set_axis_dt(t2, t0=t0).index.values

    t2_prev_t = interp1d(
        st_data_t,
        st_data_t,
        kind="previous",
        fill_value="extrapolate"
    )
    
    def R_penalized(t):
        dt = t - t2_prev_t(t)
        return R_nom * dt_cov_penalty(dt)
    
    return R_penalized



# +
def get_t2y(rdata: data_utils.RunData, corner, t0, src, inout=None):
    t2 = get_t2_data(rdata, src=src, corner=corner, inout=inout)
    
    t2y = interp1d(
        data_utils.get_dt(t2, t0=t0),
        t2.values,
        kind="previous",
        fill_value="extrapolate"
    )
    return t2y

def run_kalmanfilter(
    kf, rdata: data_utils.RunData, R_nom, corner, inout, dt, t2u_factory, ambient, T2src, x0=None,
) -> kalman.KalmanTrajectory:
    """
    Parameters
    -----------------
                
    rdata: RunData
        Experimental dataset
        
    corner: string
        One of "RR", "FR", "FL", "RL"
        
    inout: string
        One of "in", "out". Which thermocouple probe channel to use (inboard or outboard).
        
    Returns
    ----------------
    
    traj: kalman.KalmanTrajectory
    
    cdata: data_utils.RunData
        Data with cropped time, used to run the KF.
    
    """
    # Crop data starting at first smart track sample
    t2_data_uncropped_nona = get_t2_data(rdata, src=T2src, corner=corner, inout=inout)
    if len(t2_data_uncropped_nona.index) == 0:
        raise ValueError("No available T2 data")
    time_obs_start = t2_data_uncropped_nona.index[0]
    cdata = data_utils.crop_data(rdata, start=time_obs_start)
    if T2src == "smarttrack":
        assert cdata.smart_track.index[0] == time_obs_start
    elif T2src == "tms":
        assert cdata.tms.index[0] == time_obs_start
    
    t2u = t2u_factory(cdata, corner, inout, t0=time_obs_start, ambient_temp=ambient)
    t2y = get_t2y(cdata, corner, inout=inout, src=T2src, t0=time_obs_start)
    
    tf = data_utils.get_dt(cdata.malt, time_obs_start)[-1] + 1.0
    t_samples = np.arange(0, tf, dt, dtype=float)
    
    # Initialize KF estimate T1 and T2 to first smart track measurement
    if x0 is None:
        x0 = np.array([t2y(0), t2y(0)])
    kf.reset(x0)
    
    kf.set_R(get_R_fun(R_nom, cdata, T2src=T2src, corner=corner, inout=inout, t0=time_obs_start))    
    
    traj = kf.predict_trajectory(t_samples, t2u, t2y)
    return (traj, cdata)


# -

def plot_kf_traj(
    traj: kalman.KalmanTrajectory, cdata: data_utils.RunData, T2src, corner, inout
):
    t0 = cdata.smart_track.index[0]
    
    cov_upper = traj.x[:, 0] + np.sqrt(traj.cov[:, 0, 0]) * 2
    cov_lower = traj.x[:, 0] - np.sqrt(traj.cov[:, 0, 0]) * 2

    fig, axes = plt.subplots(2, 1, figsize=[11, 9], sharex=True)
    f
    axes[0].fill_between(traj.t, cov_lower, cov_upper, color="C0", alpha=0.15)    
    axes[0].plot(traj.t, traj.x[:, 0], color="C0", label="$T_1$ Estimate")
    axes[0].plot(
        data_utils.set_axis_dt(cdata.get_probe_in(corner=corner, inout=inout), t0),
        "-",
        color="C2",
        label="$T_1$ meas. (probe)"
    )

    T2_meas = get_t2_data(cdata, src=T2src, corner=corner, inout=inout)
    axes[0].plot(traj.t, traj.x[:, 1], color="C3", label="$T_2$ Estimate")
    axes[0].plot(
        data_utils.set_axis_dt(T2_meas, t0),
        ls="none",
        marker=".",
        color="C1",
        label="$T_2$ meas. (smart track)"
    )
    
    axes[0].set(ylabel="Temperature (°C)")
    axes[0].legend()

    ta = axes[1].twinx()
    axes[1].plot(data_utils.get_dt(cdata.can, t0), cdata.can.NavSpeed, label="Tractor speed")
    axes[1].set(ylabel="Speed (km/h)", xlabel="Time (s)")
    axes[1].legend(loc=(0.80, 0.9))

    fa_df = data_utils.set_axis_dt(cdata.get_malt_channel(corner=corner, inout=inout), t0)
    fa_straddle = data_utils.mask_straddle(fa_df, cdata)
    ta.plot(fa_straddle, color="C4", lw=6, label="Straddle")

    ta.plot(fa_df, color="C1", label="Axle load")
    ta.set(ylabel="Load (lbf)")
    ta.grid(False)

    ta.legend(loc=(0.8, 0.75))

    return (fig, axes)


# +
modelcls = models.TwoThermalMassGround_TempT2u_exp2

sp0 = [
    4.690596490059310497e+01,
    1.218325057952611701e+00,
    1.010263317267678929e-01,
    4.179452360352673423e+00,
    1.361680201561210168e+00,
    9.356137230537612481e+01,
    5.259428113435824059e+00,
]

nlsys = modelcls.from_optimvars(sp0, x0=None, t2u=None)

dt = 1.0

# Noise variance in input space
Qu_ekf = np.diag([100**2, 20**2, 20**2]) # Noise variance for inputs Q_in and T_0

# Project Q to state space using linearized B matrix
nlsys.xbar = np.array([120, 50]) # Equilibrium state for linearization
nlsys.ubar = np.array([0, 25, 25])   # Equilibrium input for linearization
sys_linearized = statespace.linearize(nlsys, 0.1)
Qx_ekf = sys_linearized.B @ Qu_ekf @ sys_linearized.B.transpose()
print("B linearized =\n", sys_linearized.B)
print("Qx =\n", Qx_ekf)

R_ekf = np.array(8**2).reshape(1, 1)  # Noise variance for measurement T_2

ekf = kalman.DiscreteExtKalman(
    x0=[0, 0],
    cov0=np.diag([1, 1]) * 1E-5,
    f=nlsys.f,
    h=nlsys.h,
    Q=Qx_ekf,
    R=R_ekf,
)

traj, cdata = run_kalmanfilter(
    ekf,
    rdata_5_6_7,
    R_nom=R_ekf,
    corner="RR",
    inout="IN",
    dt=dt,
    x0=[64, 43],
    t2u_factory=modelcls.get_t2u,
    T2src="smarttrack",
    ambient="ground"
)

fig, axes = plot_kf_traj(traj, cdata, corner="RR", inout="IN", T2src="smarttrack")

nlsys.x0 = np.array([64, 43])
nlsys.t2u = modelcls.get_t2u(cdata, "RR", "IN", ambient_temp="ground", t0=None)
sys_traj = nlsys.compute_trajectory(tf=traj.t[-1], n=2000, rtol=1E-4)
axes[0].plot(sys_traj.t, sys_traj.x[:, 0], color="C3")

fig2, axes2 = plt.subplots(1, 1, figsize=[11, 9])
axes2.plot(sys_traj.t, sys_traj.x[:, 0], color="C3")
plt.show()


# -

# ## EKF using smart track sensors

def eval_t1_residuals(kftraj, cropped_data, corner, inout):
    t1_probe_data = data_utils.set_axis_dt(
        cropped_data.get_probe_in(corner=corner, inout=inout),
        t0=None
    ).dropna()
    t1_kf_interp = interp1d(kftraj.t, kftraj.x[:, 0])
    return t1_kf_interp(t1_probe_data.index) - t1_probe_data


# +
sim_x0 = [
    np.array([64, 43]),
    np.array([23, 23]),
    np.array([26, 26]),
    np.array([23, 23]),# below is just a general estimation for start (always the same one)
    np.array([23, 23]),
    np.array([23, 23]),
    np.array([23, 23]),
]

modelcls = models.TwoThermalMassGround_TempT2u_exp2_alpha

sp0 = [
    2.602554306206707224e+01,
    7.203994445713135519e-02,
    1.607687548431471214e-01,
    8.056876478379166429e+01,
    2.600400033712707426e+01,
    6.841330569466666134e+01,
    1.585490161451699009e+01,
    1.791721810347513211e+00,
]

nlsys = modelcls.from_optimvars(sp0, x0=None, t2u=None)

dt = 1.0

# Noise variance in input space
Qu_ekf = np.diag([10**2, 25**2, 25**2]) # Noise variance for inputs Q_in and T_0

# Project Q to state space using linearized B matrix
nlsys.xbar = np.array([120, 50]) # Equilibrium state for linearization
nlsys.ubar = np.array([100, 25, 25])   # Equilibrium input for linearization
sys_linearized = statespace.linearize(nlsys, 0.1)
Qx_ekf = sys_linearized.B @ Qu_ekf @ sys_linearized.B.transpose()
print("B linearized =\n", sys_linearized.B)
print("Qx =\n", Qx_ekf)

R_ekf = np.array(4**2).reshape(1, 1)  # Noise variance for measurement T_2

ekf = kalman.DiscreteExtKalman(
    x0=[0, 0],
    cov0=np.diag([1, 1]) * 1E-5,
    f=nlsys.f,
    h=nlsys.h,
    Q=Qx_ekf,
    R=R_ekf,
)

all_residuals = []
for i, name in enumerate(all_run_datasets.keys()):
    rdata = all_run_datasets[name]
    
    traj, cropped_data = run_kalmanfilter(
        ekf,
        rdata,
        R_nom=R_ekf,
        corner="RR",
        inout="IN",
        dt=dt,
        x0=sim_x0[i],
        t2u_factory=nlsys.get_t2u_instance,
        ambient="ground",
        T2src="smarttrack"
    )
    
    fig, axes = plot_kf_traj(traj, cropped_data, T2src="smarttrack", corner="RR", inout="IN")
    fig.suptitle(f"Runs {name}")
    
    nlsys.x0 = sim_x0[i]
    nlsys.t2u = nlsys.get_t2u(cropped_data, "RR", "IN", t0=None, ambient_temp="ground")
    sys_traj = nlsys.compute_trajectory(tf=traj.t[-1], n=2000, rtol=1E-4)
    axes[0].plot(sys_traj.t, sys_traj.x[:, 0], color="C5", label="Model sim.")
    axes[0].legend()
    
    fname_suffix = name.replace(", ", "_")
    fig.savefig(f"C:/Users/TDewitte/Desktop/PythonPrograms/S4S/camso-s4s/observer_v2_ekf_st_{fname_suffix}.svg")

    # plot residuals
    res = eval_t1_residuals(traj, cropped_data, corner="RR", inout="in")
    fig, axes = plt.subplots(2, 1, figsize=[11, 9])
    axes[0].plot(res, marker="o", ls="none")
    axes[1].hist(res, bins=30)
    all_residuals.append(res)

all_residuals = np.concatenate(all_residuals, axis=0)
res_q = np.quantile(np.abs(all_residuals), 0.95)
print(f"Q_0.95 of residuals: {res_q}")
# -
# ### With unknown initial state

# ## EKF with TMS sensors


# +
# %matplotlib widget

position = "RR"
side = "in"

sim_x0 = [
    np.array([64, 35]),
    np.array([23, 18]),
    np.array([26, 18]),
    # np.array([23, 23]),# below is just a general estimation for start (always the same one)
    # np.array([23, 23]),
    # np.array([23, 23]),
    # np.array([23, 23])
    np.array([23, 23]),
    np.array([100, 40]),
    np.array([23, 23]),
    np.array([60, 40]),
    np.array([26, 18]),
    np.array([100, 40])
]

modelcls = models.TwoThermalMassGround_TempT2u_exp2_alpha
"""
## TMS mean and alpha not equal to 2 in 'global_optimization'
sp0 = [
    1.105700752905286954e+01,
    1.568312950498511049e-01,
    3.741747847334142008e-01,
    5.620713100072365620e+01,
    1.062964404048086386e+01,
    5.192011997531415091e+01,
    3.060138358572021033e+01,
    1.639732328657291793e+00,
]


# TMS median and alpha not equal to 2 in 'global_optimization'
sp0 = [
    4.867815311380567778e-01,
    1.128602803895844886e+00,
    1.311823588114514649e+01,
    1.573198337015163162e+01,
    2.784295977308133274e+00,
    6.805796636052055248e+01,
    4.775073734311296647e+01,
    1.291080118154863898e+00,
]
"""
# TMS minimum and alpha not equal to 2 in 'global_optimization'
sp0 = [
    4.893127917901352930e+00,  # c1=optim_vars[0] * 1000,
    1.089864217734101004e+01,  # c2=optim_vars[1] * 1000,
    8.886049019440918073e-01,  # r1g=optim_vars[2],
    4.820416805331718990e+01,  # r12=optim_vars[3],
    2.491513894780723604e-01,  # r20=optim_vars[4],
    6.047796433047486175e+01,  # delta_r100=optim_vars[5] * 0.01,
    2.801337541133028353e+01,  # delta_r200=optim_vars[6] * 0.01,
    1.560527042709535461e+00,  # alpha=optim_vars[7],
]

nlsys = modelcls.from_optimvars(sp0, x0=None, t2u=None)

dt = 1.0

# Noise variance in input space
Qu_ekf = np.diag([2**2, 500**2, 500**2]) #np.diag([10**2, 25**2, 25**2]) # Noise variance for inputs Q_in and T_0

# Project Q to state space using linearized B matrix
nlsys.xbar = np.array([120, 50]) # Equilibrium state for linearization
nlsys.ubar = np.array([100, 25, 25])   # Equilibrium input for linearization
sys_linearized = statespace.linearize(nlsys, 0.1)
Qx_ekf = sys_linearized.B @ Qu_ekf @ sys_linearized.B.transpose()
print("B linearized =\n", sys_linearized.B)
print("Qx =\n", Qx_ekf)

R_ekf = np.array(8**2).reshape(1, 1)  # Noise variance for measurement T_2

ekf = kalman.DiscreteExtKalman(
    x0=[0, 0],
    cov0=np.diag([1, 1]) * 1E-5,
    f=nlsys.f,
    h=nlsys.h,
    Q=Qx_ekf,
    R=R_ekf,
)

all_residuals = []
for i, name in enumerate(all_run_datasets.keys()):
    rdata = all_run_datasets[name]
    
    traj, cropped_data = run_kalmanfilter(
        ekf,
        rdata,
        R_nom=R_ekf,
        corner=position,
        inout=side,
        dt=dt,
        x0=sim_x0[i],
        t2u_factory=nlsys.get_t2u_instance,
        ambient="ground",
        T2src="tms"
    )
    
    fig, axes = plot_kf_traj(traj, cropped_data, T2src="tms", corner=position, inout=side)
    fig.suptitle(f"Runs {name}")
    
    nlsys.x0 = sim_x0[i]
    nlsys.t2u = nlsys.get_t2u(cropped_data, position, side, t0=None, ambient_temp="ground")
    sys_traj = nlsys.compute_trajectory(tf=traj.t[-1], n=2000, rtol=1E-4) # tf= final step, n= time steps, rtol is a solver argument
    axes[0].plot(sys_traj.t, sys_traj.x[:, 0], color="C5", label="Model sim.")
    axes[0].legend()
        
    fname_suffix = name.replace(", ", "_")
    # fig.savefig(f"C:/temp/observer_v2_ekf_tms_{fname_suffix}.svg")
    fig.savefig(f"C:/Users/TDewitte/Desktop/PythonPrograms/S4S/camso-s4s/observer_v2_ekf_tms_{fname_suffix}_{position}_{side}.svg")

    # plot residuals
    res = eval_t1_residuals(traj, cropped_data, corner=position, inout=side)
    fig, axes = plt.subplots(2, 1, figsize=[11, 9])
    axes[0].plot(res, marker="o", ls="none")
    axes[1].hist(res, bins=30)
    all_residuals.append(res)

all_residuals = np.concatenate(all_residuals, axis=0)
res_q = np.quantile(np.abs(all_residuals), 0.95)
print(f"Q_0.95 of residuals: {res_q}")
# +
# # %matplotlib tk

position = "RR"
side = "in"

sim_x0 = [
    np.array([20, 20]),  # Start temperature at ambient temperature
]

modelcls = models.TwoThermalMassGround_TempT2u_exp2_alpha

# TMS median and alpha not equal to 2 in 'global_optimization'
sp0 = [
    4.893127917901352930e+00,
    1.089864217734101004e+01,
    8.886049019440918073e-01,
    4.820416805331718990e+01,
    2.491513894780723604e-01,
    6.047796433047486175e+01,
    2.801337541133028353e+01,
    1.560527042709535461e+00,
]

nlsys = modelcls.from_optimvars(sp0, x0=None, t2u=None)

dt = 1.0

# best so far: Qu_ekf = np.diag([2**2, 50**2, 50**2]) and R_ekf = np.array(2**2).reshape(1, 1) or Qu_ekf = np.diag([1**2, 1**2, 1**2]) and R_ekf = np.array(8**2).reshape(1, 1)

# Noise variance in input space
Qu_ekf = np.diag([1**2, 1**2, 1**2]) #np.diag([2**2, 50**2, 50**2]) # np.diag([0.1, 0.1, 0.1]) #np.diag([10**2, 25**2, 25**2]) # np.diag([0.1, 0.1, 0.1])# 
# Noise variance for inputs Q_in and T_0

# Project Q to state space using linearized B matrix
nlsys.xbar = np.array([120, 50]) # Equilibrium state for linearization
nlsys.ubar = np.array([100, 25, 25])   # Equilibrium input for linearization
sys_linearized = statespace.linearize(nlsys, 0.1)
Qx_ekf = sys_linearized.B @ Qu_ekf @ sys_linearized.B.transpose()  # @ is the operator for matrix multiplication
print("B linearized =\n", sys_linearized.B)
print("Qx =\n", Qx_ekf)

R_ekf = np.array(8**2).reshape(1, 1)  #np.array(8**2).reshape(1, 1)  # Noise variance for measurement T_2

ekf = kalman.DiscreteExtKalman(
    x0=[0, 0],
    cov0=np.diag([1, 1]) * 1E-5,
    f=nlsys.f,
    h=nlsys.h,
    Q=Qx_ekf,
    R=R_ekf,
)

all_residuals = []
for i, name in enumerate(all_runs.keys()):
    rdata = all_runs[name]
    
    traj, cropped_data = run_kalmanfilter(
        ekf,
        rdata,
        R_nom=R_ekf,
        corner=position,
        inout=side,
        dt=dt,
        x0=sim_x0[i],
        t2u_factory=nlsys.get_t2u_instance,
        ambient="ground",
        T2src="tms"
    )
    
    fig, axes = plot_kf_traj(traj, cropped_data, T2src="tms", corner=position, inout=side)
    fig.suptitle(f"Runs {name}")
    
    nlsys.x0 = sim_x0[i]
    nlsys.t2u = nlsys.get_t2u(cropped_data, position, side, t0=None, ambient_temp="ground")
    sys_traj = nlsys.compute_trajectory(tf=traj.t[-1], n=4000, rtol=1E-4) # tf= final step, n= time steps, rtol is a solver argument
    axes[0].plot(sys_traj.t, sys_traj.x[:, 0], color="C5", label="Model sim.")
    axes[0].legend()
        
    fname_suffix = name.replace(", ", "_")
    # fig.savefig(f"C:/temp/observer_v2_ekf_tms_{fname_suffix}.svg")
    fig.savefig(f"C:/Users/TDewitte/Desktop/PythonPrograms/S4S/camso-s4s/observer_v2_ekf_tms_{fname_suffix}_{position}_{side}.svg")
    pickle.dump(fig, open(f'Model_{fname_suffix}_{position}_{side}.fig.pickle', 'wb')) # Save the interactive plot

    # plot residuals
    res = eval_t1_residuals(traj, cropped_data, corner=position, inout=side)
    fig, axes = plt.subplots(2, 1, figsize=[11, 9])
    axes[0].plot(res, marker="o", ls="none")
    axes[1].hist(res, bins=30)
    all_residuals.append(res)
    pickle.dump(fig, open(f'Residuals_{fname_suffix}_{position}_{side}.fig.pickle', 'wb')) # Save the interactive plot

all_residuals = np.concatenate(all_residuals, axis=0)
res_q = np.quantile(np.abs(all_residuals), 0.95)
print(f"Q_0.95 of residuals: {res_q}")
# -
# ## TMS as ambient, 2DOF (but in fact only 1 DOF)
#
# List of parameters:
#
# c1=optim_vars[0] * 1000
#
# c2=optim_vars[1] * 1000
#
# r1g=optim_vars[2]
#
# r12=optim_vars[3]
#
# r20=optim_vars[4]
#
# delta_r100=optim_vars[5] * 0.01
#
# delta_r200=optim_vars[6] * 0.01
#
# alpha=optim_vars[7]

# +
# TMS as ambient, 2DOF, TwoThermalMassGround_TempT2u_exp2_alpha

position = "RR"
side = "in"

sim_x0 = [
    # np.array([20, 20]),  # Start temperature at ambient temperature
    np.array([64, 35]),
    np.array([23, 18]),
    np.array([26, 18]),
]

modelcls = models.TwoThermalMassGround_TempT2u_exp2_alpha

# TMS median and alpha not equal to 2 in 'global_optimization'
sp0 = [
    #2.027084024788468852e+01,  # With runs 29, 30, 31
    #6.183739988894454598e+01,
    #2.044401671385216446e-01,
    #2.800245593647997921e+01,
    #3.517105909557280086e+01,
    #4.563527349893635687e+01,
    #2.695310501766443068e+01,
    #1.704335540430848450e+00

    1.042927721362147508e+00,  # With runs 11-14
    6.447366148824112031e+01,
    5.231245988187545493e+01,
    3.391538710220721953e+00,
    7.448514847872378652e+01,
    4.946044676030431475e+01,
    4.649232447250269473e+01,
    1.377562841112280667e+00
]

nlsys = modelcls.from_optimvars(sp0, x0=None, t2u=None)

dt = 1.0

# best so far: Qu_ekf = np.diag([2**2, 50**2, 50**2]) and R_ekf = np.array(2**2).reshape(1, 1) or Qu_ekf = np.diag([1**2, 1**2, 1**2]) and R_ekf = np.array(8**2).reshape(1, 1)

# Noise variance in input space
Qu_ekf = np.diag([1**2, 1**2, 1**2]) #np.diag([2**2, 50**2, 50**2]) # np.diag([0.1, 0.1, 0.1]) #np.diag([10**2, 25**2, 25**2]) # np.diag([0.1, 0.1, 0.1])# 
# Noise variance for inputs Q_in and T_0

# Project Q to state space using linearized B matrix
nlsys.xbar = np.array([120, 50]) # Equilibrium state for linearization
nlsys.ubar = np.array([100, 25, 25])   # Equilibrium input for linearization
sys_linearized = statespace.linearize(nlsys, 0.1)
Qx_ekf = sys_linearized.B @ Qu_ekf @ sys_linearized.B.transpose()  # @ is the operator for matrix multiplication
print("B linearized =\n", sys_linearized.B)
print("Qx =\n", Qx_ekf)

R_ekf = np.array(8**2).reshape(1, 1)  #np.array(8**2).reshape(1, 1)  # Noise variance for measurement T_2

ekf = kalman.DiscreteExtKalman(
    x0=[0, 0],
    cov0=np.diag([1, 1]) * 1E-5,
    f=nlsys.f,
    h=nlsys.h,
    Q=Qx_ekf,
    R=R_ekf,
)

all_residuals = []
for i, name in enumerate(all_run_datasets.keys()):
    rdata = all_run_datasets[name]
    
    traj, cropped_data = run_kalmanfilter(
        ekf,
        rdata,
        R_nom=R_ekf,
        corner=position,
        inout=side,
        dt=dt,
        x0=sim_x0[i],
        t2u_factory=nlsys.get_t2u_instance,
        ambient="TMS",
        T2src="tms"
    )
    
    fig, axes = plot_kf_traj(traj, cropped_data, T2src="tms", corner=position, inout=side)
    fig.suptitle(f"Runs {name}")
    
    nlsys.x0 = sim_x0[i]
    nlsys.t2u = nlsys.get_t2u(cropped_data, position, side, t0=None, ambient_temp="TMS")
    sys_traj = nlsys.compute_trajectory(tf=traj.t[-1], n=4000, rtol=1E-4) # tf= final step, n= time steps, rtol is a solver argument
    axes[0].plot(sys_traj.t, sys_traj.x[:, 0], color="C5", label="Model sim.")
    axes[0].legend()
        
    fname_suffix = name.replace(", ", "_")
    # fig.savefig(f"C:/temp/observer_v2_ekf_tms_{fname_suffix}.svg")
    fig.savefig(f"C:/Users/TDewitte/Desktop/PythonPrograms/S4S/camso-s4s/observer_v2_ekf_tms_{fname_suffix}_{position}_{side}.svg")
    pickle.dump(fig, open(f'Model_{fname_suffix}_{position}_{side}.fig.pickle', 'wb')) # Save the interactive plot

    # plot residuals
    res = eval_t1_residuals(traj, cropped_data, corner=position, inout=side)
    fig, axes = plt.subplots(2, 1, figsize=[11, 9])
    axes[0].plot(res, marker="o", ls="none")
    axes[1].hist(res, bins=30)
    all_residuals.append(res)
    pickle.dump(fig, open(f'Residuals_{fname_suffix}_{position}_{side}.fig.pickle', 'wb')) # Save the interactive plot

all_residuals = np.concatenate(all_residuals, axis=0)
res_q = np.quantile(np.abs(all_residuals), 0.95)
print(f"Q_0.95 of residuals: {res_q}")
# +
# # %matplotlib tk

position = "RR"  # !!!!!!!!!  to adjust in the 'get_ambient_temp_tms' (data_utils) as well   !!!!!!!!!!!!!!!
side = "in"

sim_x0 = [
    np.array([20, 20]),  # Start temperature at ambient temperature
]

modelcls = models.TwoThermalMassGround_TempT2u_exp2_alpha

# TMS median and alpha not equal to 2 in 'global_optimization'
sp0 = [
    #2.027084024788468852e+01,  # With runs 29, 30, 31
    #6.183739988894454598e+01,
    #2.044401671385216446e-01,
    #2.800245593647997921e+01,
    #3.517105909557280086e+01,
    #4.563527349893635687e+01,
    #2.695310501766443068e+01,
    #1.704335540430848450e+00

    1.042927721362147508e+00,  # With runs 11-14
    6.447366148824112031e+01,
    5.231245988187545493e+01,
    3.391538710220721953e+00,
    7.448514847872378652e+01,
    4.946044676030431475e+01,
    4.649232447250269473e+01,
    1.377562841112280667e+00
]

nlsys = modelcls.from_optimvars(sp0, x0=None, t2u=None)

dt = 1.0

# best so far: Qu_ekf = np.diag([2**2, 50**2, 50**2]) and R_ekf = np.array(2**2).reshape(1, 1) or Qu_ekf = np.diag([1**2, 1**2, 1**2]) and R_ekf = np.array(8**2).reshape(1, 1)

# Noise variance in input space
Qu_ekf = np.diag([1**2, 1**2, 1**2]) #np.diag([2**2, 50**2, 50**2]) # np.diag([0.1, 0.1, 0.1]) #np.diag([10**2, 25**2, 25**2]) # np.diag([0.1, 0.1, 0.1])# 
# Noise variance for inputs Q_in and T_0

# Project Q to state space using linearized B matrix
nlsys.xbar = np.array([120, 50]) # Equilibrium state for linearization
nlsys.ubar = np.array([100, 25, 25])   # Equilibrium input for linearization
sys_linearized = statespace.linearize(nlsys, 0.1)
Qx_ekf = sys_linearized.B @ Qu_ekf @ sys_linearized.B.transpose()  # @ is the operator for matrix multiplication
print("B linearized =\n", sys_linearized.B)
print("Qx =\n", Qx_ekf)

R_ekf = np.array(8**2).reshape(1, 1)  #np.array(8**2).reshape(1, 1)  # Noise variance for measurement T_2

ekf = kalman.DiscreteExtKalman(
    x0=[0, 0],
    cov0=np.diag([1, 1]) * 1E-5,
    f=nlsys.f,
    h=nlsys.h,
    Q=Qx_ekf,
    R=R_ekf,
)

all_residuals = []
for i, name in enumerate(all_runs.keys()):
    rdata = all_runs[name]
    
    traj, cropped_data = run_kalmanfilter(
        ekf,
        rdata,
        R_nom=R_ekf,
        corner=position,
        inout=side,
        dt=dt,
        x0=sim_x0[i],
        t2u_factory=nlsys.get_t2u_instance,
        ambient="TMS",
        T2src="tms"
    )
    
    fig, axes = plot_kf_traj(traj, cropped_data, T2src="tms", corner=position, inout=side)
    fig.suptitle(f"Runs {name}")
    
    nlsys.x0 = sim_x0[i]
    nlsys.t2u = nlsys.get_t2u(cropped_data, position, side, t0=None, ambient_temp="TMS")
    sys_traj = nlsys.compute_trajectory(tf=traj.t[-1], n=4000, rtol=1E-4) # tf= final step, n= time steps, rtol is a solver argument
    axes[0].plot(sys_traj.t, sys_traj.x[:, 0], color="C5", label="Model sim.")
    axes[0].legend()
        
    fname_suffix = name.replace(", ", "_")
    # fig.savefig(f"C:/temp/observer_v2_ekf_tms_{fname_suffix}.svg")
    fig.savefig(f"C:/Users/TDewitte/Desktop/PythonPrograms/S4S/camso-s4s/observer_v2_ekf_tms_{fname_suffix}_{position}_{side}.svg")
    pickle.dump(fig, open(f'Model_{fname_suffix}_{position}_{side}.fig.pickle', 'wb')) # Save the interactive plot

    # plot residuals
    res = eval_t1_residuals(traj, cropped_data, corner=position, inout=side)
    fig, axes = plt.subplots(2, 1, figsize=[11, 9])
    axes[0].plot(res, marker="o", ls="none")
    axes[1].hist(res, bins=30)
    all_residuals.append(res)
    pickle.dump(fig, open(f'Residuals_{fname_suffix}_{position}_{side}.fig.pickle', 'wb')) # Save the interactive plot

all_residuals = np.concatenate(all_residuals, axis=0)
res_q = np.quantile(np.abs(all_residuals), 0.95)
print(f"Q_0.95 of residuals: {res_q}")
# -

