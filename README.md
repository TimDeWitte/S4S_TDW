# S4S_TDW

I played around (a lot) with all different options and made sure the TMS can be used as an ambient temperature and as T2 at the same time (bottom parts of observer_v2.py). Then I also tried a new approach after figuring out that the 2DOF system is in fact a 1DOF system. When I tried my new 1DOF (and the old 2DOF) model in other corners (RR_out, FR_in, FR_out), it was notoriously bad. 

Now I am trying to come up with a simple 1 DOF model and I am looking at the other available positions already from the beginning. No real final model yet, but I am getting there. The two python files starting with 'TDW' are what I have made so far for the 1DOF simple model. There I am basically trying to come of with something that always envelops the worst case, as I also know that the probe measurements are not perfect and  not always on the exact location of the hottest point. They are in fact always 'underestimating' and they will never measure higher than the real hotspot.
