# +
import numpy as np

from scipy import optimize

import scipy.special

from matplotlib import pyplot as plt

from time import perf_counter_ns

plt.style.use("ggplot")
# %matplotlib widget

# +
tt = np.linspace(0, 200, 100)

gtd_lin = 1 - 5E-3 * tt 

fig, ax = plt.subplots()
ax.plot(tt, gtd_lin, label="Linear")

xi = 10
xf = 0
b = -0.01
a = (xf - xi) / (np.exp(200 * b) - 1)
c = xi - a
print(f"a={a}, b={b}, c={c}")

gtd_exp = a * np.exp(tt*b) + c
ax.plot(tt, gtd_exp, label="Exponential")
ax.axhline(gtd_exp[-1], ls="--", color="grey", alpha=0.2)

gtd_sqrt = 1 - np.sqrt(0.003 * tt)
ax.plot(tt, gtd_sqrt, label="Square root")

gtd_poly = 1 / ((tt + 273) * 0.0001) / 35
ax.plot(tt, gtd_poly, label="inverse square")

#gtd_log = 1 - np.log(tt) / 5
#ax.plot(tt, gtd_log, label="log")

ax.legend()
# +
# Intuitive parameterization for 3-parameter exponential function
# Intuitive paramaters are the desired function are:
# d0 = value at 0 deg
# d200 = value at 200 deg / value at 0 deg
# d100 = value at 100 deg as linear ratio between d0 and d200

# d100 controls curvature, should be in open interval ]0, 1[
# d100 = 0.5 means linear

d0 = 100
d200 = 0.5
d100 = 0.1

def exp_param_cost(b):
    val_0 = d0
    val_200 = d0 * d200
    val_100_target = d0 + (val_200 - d0) * d100
    
    a = (val_200 - val_0) / (np.exp(200 * b) - 1)
    c = d0 - a
    val_100 = a * np.exp(b * 100) + c
    return (val_100 - val_100_target) ** 2

tic = perf_counter_ns()
optimres = optimize.minimize_scalar(exp_param_cost, (-0.1, 0.1))
toc = perf_counter_ns()
print(f"Elapsed = {(toc - tic) / 1E6} ms")

print(optimres)
b = optimres.x
val_0 = d0
val_200 = d0 * d200
a = (val_200 - val_0) / (np.exp(200 * b) - 1)
c = d0 - a

print(f"a={a:.6e} b={b:.6e} c={c:.6e}")

fig, ax = plt.subplots()
ax.plot(tt, (a*np.exp(b*tt)+c))
# -
a


# +
def exp_find_abc(y0, r_y100, r_y200):
    d = 100
    v1 = y0
    v3 = v1 * r_y200
    v2 = v1 + (v3 - v1) * r_y100

    r = (v3 - v2) / (v2 - v1)
    b = np.log(r) / d
    a = (v2 - v1) / (r - 1)
    c = y0 - a
    return np.array([a, b, c])

a, b, c = exp_find_abc(100, 0.6, 0.5)
print(b)

fig, ax = plt.subplots()
ax.plot(tt, (a*np.exp(b*tt)+c))
# +
fig, ax = plt.subplots()

for r100 in [0.01, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 0.99]:
    a, b, c = exp_find_abc(100, r100, 0.5)
    ax.plot(tt, (a*np.exp(b*tt)+c), color="C0")

for r100 in [0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75]:
    poly = np.polynomial.Polynomial.fit(
        [0, 100, 200],
        [100, (100 - r100 * 50), 50],
        2
    )
    ax.plot(tt, poly(tt), color="C1")
# -


