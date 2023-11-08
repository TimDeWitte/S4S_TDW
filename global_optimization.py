import sys

import pickle

import numpy as np

from scipy.optimize import differential_evolution

import models

import optim_utils

with open("./concat_data/runs_5_6_7.pickle", "rb") as f:
    rdata_5_6_7 = pickle.load(f)

with open("./concat_data/runs_29_30_31.pickle", "rb") as f:
    rdata_29_30_31 = pickle.load(f)

with open("./concat_data/runs_18_19_20.pickle", "rb") as f:
    rdata_18_19_20 = pickle.load(f)

# added on 2/11/23 TDW
with open("./concat_data/runs_11_12_13_14.pickle", "rb") as f:
    rdata_11_12_13_14 = pickle.load(f)


# ## Global Optimization

def globalopt_model_multiple(
    modelcls,
    rdata,
    corner,
    inout,
    bounds,
    ambient_temp,
    T2src,  # T2  source (can be smart track or TMS or ...)
    t1_init=None,
    t2_init=None,
    init_params=None,
    logfile=None,
):
    optim_cost_fun = optim_utils.ModelOptimizationCostFunctionNSS(
        modelcls, rdata, corner, inout, ambient_temp, T2src, t1_init, t2_init
    )

    def cb(xk, convergence):
        cost = optim_cost_fun(xk)
        entry = f"cost: {cost:.2f} convergence: {convergence:.3f}"
        print(xk)
        print(entry)
        if logfile is not None:
            logfile.write(entry)
            logfile.write("\n")
            np.savetxt(logfile, xk)
            logfile.write("\n")
            logfile.flush()  # To force all buffered output to a particular log file, use the flush command. This lets you examine a current log file offline while the normal log file is still open.

    print(f"Init params: {init_params}")
    if init_params is not None:
        print(f"Initial cost = {optim_cost_fun(init_params)}")

    optimres = differential_evolution(
        optim_cost_fun,
        bounds=bounds,
        polish=False,
        callback=cb,
        x0=init_params,
        tol=0.1,
        mutation=(0.5, 1.2),
        recombination=0.7,
        init="sobol",
        maxiter=2_000,
        workers=8,
    )

    if logfile is not None:
        logfile.write(repr(optimres) + "\n")
        logfile.write("x = \n")
        np.savetxt(logfile, optimres.x)
        logfile.flush()

    return optimres


if __name__ == "__main__":
    optimvar_bounds_nl = [
        (1e-4, 100),  # c1
        (1e-4, 100),  # c2
        (1e-4, 100),  # r1g
        (1e-4, 100),  # r12
        (1e-4, 100),  # r20
        (0.1, 99.9),  # delta_r100
        # (0, 100),  # delta_r200  # swapped 17/10/23
        (0, 100),     # delta_r200  # swapped 17/10/23
        (1, 3),       # alpha  # swapped 17/10/23
    ]

    with open("globalopt_tms_exp2_gndgnd_log.txt", "wt") as f:
        optimres_global = globalopt_model_multiple(
            models.TwoThermalMassGround_TempT2u_exp2_alpha, #_exp2, # swapped 17/10/23
            rdata=[rdata_5_6_7, rdata_18_19_20, rdata_11_12_13_14], # rdata_29_30_31], ## Use another set to build the model
            corner="RR",
            inout="IN",
            bounds=optimvar_bounds_nl,
            #init_params=sp0,

            # Initial conditions for smarttrack sensors
            #t1_init=[None, 26, None],
            #t2_init=[55, None, 22],
            
            # Initial conditions for TMS
            t1_init=[None, 26, 19], # None],
            t2_init=[47, 20, 19],

            ambient_temp="TMS",
            T2src="tms",
            logfile=f,
        )

    np.savetxt(sys.stdout, optimres_global.x)
    print(optimres_global)


