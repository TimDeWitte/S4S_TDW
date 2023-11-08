import sys

import pickle

import numpy as np

from scipy.optimize import differential_evolution

import models

import optim_utils

import pandas as pd

from matplotlib import pyplot as plt

with open("./concat_data/runs_1-5.pickle", "rb") as f:
    data = pickle.load(f)


# # Model

def collect_runs(data, corner, inout):
    tms_col = f"tms_int_{corner.upper()}_{inout.lower()}"
    run_data = pd.DataFrame(
        {
            "speed": data.can.NavSpeed,
            "load": data.get_malt_channel(corner=corner, inout=inout),
            "tms": data.can[tms_col],
            "probe": data.get_probe_in(corner=corner, inout=inout),
        }
    )
    return run_data



# +
corner = ["RR", "RR", "FR", "FR"]
inout = ["in", "out", "in", "out"]

run_data = {}

for cornerindex in range(len(corner)):
    tms_col = f"{corner[cornerindex].upper()}_{inout[cornerindex].lower()}"
    run_data[tms_col] = collect_runs(data, corner[cornerindex], inout[cornerindex])

# print(run_data)
# -

def func_fitting(param):
    alpha = param[0]
    d0= param[1]
    d1= param[2]
    #c0= param[3]
    #c1= param[4]
    r0= param[3]
    a_1= param[4]
    T0=param[5]

    if -d0/200>d1 or -alpha/200>a_1:
        cost_max=10**30
    else:
        cost=np.zeros(len(runs))
        cost2=np.zeros(len(runs))
        for j in range(0,len(runs)):
            run_down = runs[j]
            
            v = run_down['speed'].to_numpy()
            F = run_down['load'].to_numpy()
            T = run_down['tms'].to_numpy()
            P = run_down['probe'].to_numpy()
        
            T1 = T*0
            T1[0]=T0

            cost[j]=0
        
            for i in range(1,len(v)):
                #T1[i]=T1[i-1]+(v[i]*np.abs(F[i]/1000)**alpha*(d1*T1[i-1]+d0)-T1[i-1]/(r0+r1*T1[i-1])+T[i]/(r0+r1*T1[i-1]))/(c1*T1[i-1]+c0)
                #T1[i]=T1[i-1]+(v[i]*np.abs(F[i]/1000)**alpha*(d1*T1[i-1]+d0)-T1[i-1]/(r0)+T[i]/(r0))/(c1*T1[i-1]+c0)
                #T1[i]=T1[i-1]+(v[i]*np.abs(F[i]/1000)**alpha*(d1*T1[i-1]+d0)-(T1[i-1]-T[i])/(r0+r1*T1[i-1]))
                T1[i]=T1[i-1]+(v[i]*np.abs(F[i]/1000)**(alpha+a_1*F[i]/1000)*(d1*T1[i-1]+d0)-(T1[i-1]-T[i])/(r0))
                if not np.isnan(T1[i]-P[i]):
                    if T1[i]>P[i]:
                        cost[j]=cost[j]+(T1[i]-P[i])**2
                    else: # When the prediction is smaller than the probe, we give an additional penalty. The measured hotspot temperature is always lower or equal to the real one
                        cost[j]=cost[j]+(np.abs(P[i]-T1[i])**2)*100
                i+=1
    
            #if np.nanmax(T1-P) > 150:
            #    cost[j]=10**30
            #else:
            #cost[j]=np.sqrt(np.nansum((T1-P)**2))
            #cost2[j]=np.nanmax(T1)
            if cost[j]==0:
                cost[j]=10**30
        max1=np.sqrt(np.mean(cost**2))
        max2=np.max(cost2)
        #print(max1)
        #print(max2)
        
        cost_max= np.max([max1, max2])
    #print(cost_max)
    
    return cost_max


# +
run = run_data['RR_in']

# run_down = run.resample('1S').mean()

# I downsample it all for now to make it run a lot faster
#runs = [run_data['RR_in'].resample('10S').mean(), run_data['RR_out'].resample('10S').mean(), run_data['FR_in'].resample('10S').mean(),run_data['FR_out'].resample('10S').mean()]
runs = [run_data['RR_in'].resample('10S').mean(), run_data['FR_in'].resample('10S').mean()]
print(runs)

for i in range(0,len(runs)):
    print(i)

#print(runs[2])

#print(run)
#print(run_down)

sp = (1,2,1,-1,-1,1,0.5)
cost = func_fitting(sp)

print(cost)
#plt.plot(cost)
#plt.show()

# -

# ## Global Optimization

def optimization_model(
    # rdata,
    bounds,
    init_params=None,
    logfile=None,
):

    #function_to_optimize = func_fitting(rdata, )
    
    def cb(xk, convergence):
        cost = func_fitting(xk)
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
        print(f"Initial cost = {func_fitting(init_params)}")

    # args = rdata
    optimres = differential_evolution(
        func_fitting,
        bounds=bounds,
        polish=False,
        callback=cb,
        x0=init_params,
        tol=0.1,
        mutation=(0.5, 1.2),
        recombination=0.7,
        init="sobol",
        maxiter=2_000,
        #workers=-1, # All available ones
    )

    if logfile is not None:
        logfile.write(repr(optimres) + "\n")
        logfile.write("x = \n")
        np.savetxt(logfile, optimres.x)
        logfile.flush()

    return optimres


if __name__ == "__main__":

    print('start')
    optimvar_bounds_nl = [
        (0.3, 2.2),  # alpha
        (1e-7, 100),  # delta0
        (-1e-4, 1e-4),  # delta1
        #(1e-4, 10000),  # c0
        #(-1e-10, 1e-10),  # c1
        (10, 1000),  # r0
        (-1e-2, 1e-2),  # r1, a_1
        (10, 35),   # T initial # r1
    ]

    sp0 = [2, 50, 0, 120, -0.01, 24]
    
    with open("own_model_log.txt", "wt") as f:
        optimres_global = optimization_model(
            # func_fitting,
            # rdata=run_down,
            bounds=optimvar_bounds_nl,
            init_params=sp0,
            logfile=f,
        )

    np.savetxt(sys.stdout, optimres_global.x)
    print(optimres_global)


