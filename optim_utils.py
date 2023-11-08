import numpy as np

from scipy.interpolate import interp1d

import data_utils


class ModelOptimizationCostFunction:
    def __init__(
        self, modelcls, rdata, corner, inout, ambient_temp, T2src, t1_init, t2_init
    ) -> None:
        self.n_data = len(rdata)
        self.modelcls = modelcls

        if t1_init is None:
            t1_init = [None] * self.n_data
        if t2_init is None:
            t2_init = [None] * self.n_data

        self.t1_list = []
        self.t2_list = []
        self.t2u_list = []
        self.x0_list = []
        for i in range(self.n_data):
            t0 = self.get_time_start(rdata[i], corner, inout)
            t1_raw, t2_raw = self.get_state_data(
                rdata[i], corner, inout, t0, T2src=T2src
            )
            t1, t2 = self.downsample_state_data(t1_raw, t2_raw)
            self.t1_list.append(t1)
            self.t2_list.append(t2)

            self.t2u_list.append(
                modelcls.get_t2u(
                    rdata[i], corner, inout, ambient_temp=ambient_temp, t0=t0
                )
            )

            x0 = self.get_initial_state(t1_raw, t2_raw)
            if t1_init[i] is not None:
                x0[0] = t1_init[i]
            if t2_init[i] is not None:
                x0[1] = t2_init[i]
            assert np.all(np.isfinite(x0))
            self.x0_list.append(x0)

    @staticmethod
    def get_time_start(rdata: data_utils.RunData, corner: str, inout: str):
        probe_col = f"probe_{corner.upper()}_{inout.upper()}"
        smart_track_col = corner.upper()

        return np.min(
            [
                rdata.probe[probe_col].dropna().index.values[0],
                rdata.smart_track[smart_track_col].dropna().index.values[0],
            ]
        )

    @staticmethod
    def get_state_data(
        rdata: data_utils.RunData, corner: str, inout: str, t0: np.datetime64, T2src
    ):
        """Extract T1 and T2 samples from experimental data for the purpose of
        calibrating thermal models.

        Parameters
        --------------

        rdata: RunData
            Experimental dataset

        corner: string
            One of "RR", "FR", "FL", "RL"

        inout: string
            One of "in", "out". Which thermocouple probe channel to use (inboard or outboard).

        T2src: string
            "smart_track" or "tms"

        Returns
        ------------
        t1: pandas.Series
            T1 (hotspot) samples with dt index
        t2: pandas.Series
            T2 samples with dt index
        """

        # T1 from TC probe
        probe_col = f"probe_{corner.upper()}_{inout.upper()}"
        t1 = data_utils.set_axis_dt(rdata.get_probe_in(), t0=t0)[probe_col].dropna()

        if T2src == "smarttrack":
            # T2 from smart track
            t2 = rdata.smart_track[corner.upper()].dropna()
        elif T2src == "tms":
            t2 = rdata.get_tms_channels(corner=corner, inout=inout, mean=3).dropna()  # change mean in True/False, 2, 3,... This is to be changed for e.g. mean, median, minimum,...
        else:
            raise ValueError(f"Invalid T2src option {T2src}")
        t2 = data_utils.set_axis_dt(t2, t0=t0)

        return (t1, t2)

    @staticmethod
    def downsample_state_data(t1, t2, t1_ds_factor=20):
        t1 = t1[::20]
        t2_downsample_factor = int(t2.shape[0] / t1.shape[0])
        if t2_downsample_factor > 1:
            t2 = t2[::t2_downsample_factor]
        return (t1, t2)

    @staticmethod
    def get_initial_state(t1_raw, t2_raw):
        """t1_raw, t2_raw: state samples that are NOT downsampled"""
        x0 = np.empty(2)

        # Average values in the first "x0_seconds" seconds
        x0_seconds = 20
        init_vals = [t1_raw[:x0_seconds], t2_raw[:x0_seconds]]

        for i in range(2):
            if len(init_vals[i]) > 0:
                x0[i] = init_vals[i].values.mean()
            else:
                x0[i] = np.nan

        return x0

    @staticmethod
    def eval_residuals(traj, t1_df, t2_df):
        interp_t1 = interp1d(traj.t, traj.x[:, 0], fill_value=0, bounds_error=False)(
            t1_df.index.values
        )
        interp_t2 = interp1d(traj.t, traj.x[:, 1], fill_value=0, bounds_error=False)(
            t2_df.index.values
        )
        res_t1 = t1_df.values - interp_t1
        res_t2 = t2_df.values - interp_t2
        return (res_t1, res_t2)

    def __call__(self, params) -> float:
        residuals = []
        for i in range(self.n_data):
            t2u = self.t2u_list[i]
            x0 = self.x0_list[i]
            sys = self.modelcls.from_optimvars(params, t2u, x0)

            t1, t2 = self.t1_list[i], self.t2_list[i]
            tf_sim = max([t1.index.values.max(), t2.index.values.max()]) + 60
            traj = sys.compute_trajectory(tf=tf_sim, n=200, rtol=1e-4)
            residuals.extend(self.eval_residuals(traj, t1, t2))
        residuals_arr = np.concatenate(residuals, axis=0)
        assert residuals_arr.ndim == 1
        return residuals_arr


class ModelOptimizationCostFunctionNSS(ModelOptimizationCostFunction):
    """Normalized sum of residual squares"""

    @staticmethod
    def _nss(res):
        return (res**2).sum() / res.shape[0]

    def __call__(self, params) -> float:
        residuals = super().__call__(params)
        assert residuals.ndim == 1
        return self._nss(residuals)

    @classmethod
    def eval_cost_NSS(cls, traj, t1, t2):
        residuals = ModelOptimizationCostFunction.eval_residuals(traj, t1, t2)
        residuals = np.concatenate(residuals, axis=0)
        assert residuals.ndim == 1
        return cls._nss(residuals)
