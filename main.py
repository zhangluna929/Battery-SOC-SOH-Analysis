
from __future__ import annotations
import json
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import inv
from math import sqrt
from pathlib import Path
from typing import Tuple, Dict, List, Any


np.random.seed(42)
plt.style.use("seaborn-v0_8-whitegrid")

COLORS = {
    "true": "black",
    "cc":   "tab:blue",
    "cc_ocv": "tab:orange",
    "ekf":  "tab:green",
    "ukf":  "tab:red"
}

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

class CycleData:
    """存储单条工况（电流、电压、温度）"""
    def __init__(self, current: np.ndarray, dt: float, capacity: float):
        self.current = current
        self.dt = dt
        self.capacity = capacity
        self.t = np.arange(len(current))*dt
        self.soc_true = None  # to be calculated later
        self.voltage = None
        self.temperature = np.full_like(current, 298.15)

    def add_true_soc(self, soc0: float):
        soc = soc0 - np.cumsum(self.current* self.dt / 3600)/self.capacity
        self.soc_true = np.clip(soc, 0, 1)

    def add_voltage(self, ocv_func):
        assert self.soc_true is not None, "Run add_true_soc first"
        self.voltage = ocv_func(self.soc_true) + np.random.normal(0, 0.005, len(self.soc_true))


_ocv_soc_pts = np.array([
    [0.0, 3.0],
    [0.1, 3.2],
    [0.2, 3.3],
    [0.4, 3.5],
    [0.6, 3.65],
    [0.8, 3.8],
    [1.0, 4.1],
])

from scipy.interpolate import interp1d
_ocv_interp = interp1d(_ocv_soc_pts[:,0], _ocv_soc_pts[:,1], kind='cubic')

def ocv_from_soc(soc: np.ndarray) -> np.ndarray:
    return _ocv_interp(soc)


N_STEPS = 3600  # 1h, dt=1s
pattern = np.concatenate([
    np.full(900, -4),
    np.full(600,   2),
    np.full(2100, -1.5)
])
pattern += np.random.normal(0, 0.05, N_STEPS)
cycle = CycleData(pattern, dt=1.0, capacity=2.5)
cycle.add_true_soc(0.8)
cycle.add_voltage(ocv_from_soc)

def coulomb_counting(cycle: CycleData, soc0: float) -> np.ndarray:
    soc = np.zeros_like(cycle.current)
    soc[0] = soc0
    for k in range(1, len(cycle.current)):
        soc[k] = soc[k-1] - cycle.current[k]*cycle.dt/3600/cycle.capacity
    return np.clip(soc, 0, 1)

def cc_with_ocv_correction(cycle: CycleData, soc_cc: np.ndarray) -> np.ndarray:
    soc_adj = soc_cc.copy()
    for k in range(0, len(soc_cc), 300):
        ocv_meas = cycle.voltage[k]
        soc_range = np.linspace(0, 1, 10001)
        ocv_range = ocv_from_soc(soc_range)
        idx = np.argmin(np.abs(ocv_range - ocv_meas))
        soc_adj[k:] += soc_range[idx] - soc_adj[k]
    return np.clip(soc_adj, 0, 1)


def ekf(cycle: CycleData, soc0: float, Q: float=1e-5, R: float=1e-4) -> np.ndarray:
    N = len(cycle.current)
    soc_hat = np.zeros(N)
    soc_hat[0] = soc0
    P = 1e-3
    for k in range(1, N):
        # predict
        soc_pred = soc_hat[k-1] - cycle.current[k]*cycle.dt/3600/cycle.capacity
        soc_pred = np.clip(soc_pred, 0, 1)
        P_pred = P + Q
        # observe

        s_safe = np.clip(soc_pred, 1e-6, 1 - 1e-6)
        H = (ocv_from_soc(s_safe + 1e-6) - ocv_from_soc(s_safe - 1e-6)) / 2e-6
        K = P_pred*H / (H*P_pred*H + R)
        soc_hat[k] = soc_pred + K*(cycle.voltage[k] - ocv_from_soc(soc_pred))
        P = (1 - K*H)*P_pred
    return soc_hat

class UKF:
    def __init__(self, Q: float, R: float):
        self.Q = Q; self.R = R
        self.alpha = 1e-3; self.beta = 2; self.kappa = 0
        self.n = 1  # state dim
        self.lmbd = self.alpha**2*(self.n + self.kappa) - self.n
        self.Wm = np.array([self.lmbd/(self.n+self.lmbd)] + [1/(2*(self.n+self.lmbd))]*2*self.n)
        self.Wc = self.Wm.copy(); self.Wc[0] += 1-self.alpha**2 + self.beta

    def predict(self, x, P, u):
        # state function: x - I*dt/C
        x_pred = x - u
        P_pred = P + self.Q
        return x_pred, P_pred

    def update(self, x_pred, P_pred, z_meas):
        # sigma points
        S = sqrt(self.n + self.lmbd) * np.sqrt(P_pred)
        sigmas = np.array([x_pred, x_pred + S, x_pred - S])

        sigmas_safe = np.clip(sigmas, 1e-6, 1 - 1e-6)

        zs = ocv_from_soc(sigmas_safe)
        z_pred = np.dot(self.Wm, zs)
        P_zz = self.R + np.dot(self.Wc, (zs - z_pred) ** 2)
        P_xz = np.dot(self.Wc, (sigmas_safe - x_pred) * (zs - z_pred))
        K = P_xz / P_zz

        x_new = x_pred + K * (z_meas - z_pred)
        P_new = P_pred - K * P_zz * K
        return x_new, P_new


def ukf(cycle: CycleData, soc0: float) -> np.ndarray:
    ukf_filter = UKF(Q=1e-5, R=1e-4)
    N = len(cycle.current)
    soc = np.zeros(N); soc[0] = soc0
    P = 1e-3
    for k in range(1, N):
        u = cycle.current[k]*cycle.dt/3600/cycle.capacity
        x_pred, P_pred = ukf_filter.predict(soc[k-1], P, u)
        soc[k], P = ukf_filter.update(x_pred, P_pred, cycle.voltage[k])
        soc[k] = np.clip(soc[k], 0, 1)
    return soc

def soh_capacity_fade(cycle: CycleData, soc_true: np.ndarray) -> float:
    discharge_Ah = np.sum(-cycle.current[cycle.current < 0] * cycle.dt) / 3600
    return 1 - discharge_Ah / cycle.capacity

def soh_internal_resistance(zero_r: float, r_now: float) -> float:
    return zero_r / r_now

soc_cc  = coulomb_counting(cycle, 0.8)
soc_adj = cc_with_ocv_correction(cycle, soc_cc)
soc_ekf = ekf(cycle, 0.8)
soc_ukf = ukf(cycle, 0.8)

plt.figure(figsize=(7,4))
plt.plot(cycle.t/60, cycle.soc_true, color=COLORS['true'], label="True")
plt.plot(cycle.t/60, soc_cc,      '--', color=COLORS['cc'], label="CC")
plt.plot(cycle.t/60, soc_adj,     ':', color=COLORS['cc_ocv'], label="CC+OCV")
plt.plot(cycle.t/60, soc_ekf,     '-', color=COLORS['ekf'], alpha=0.7, label="EKF")
plt.plot(cycle.t/60, soc_ukf,     '-', color=COLORS['ukf'], alpha=0.7, label="UKF")
plt.xlabel("Time / min"); plt.ylabel("SOC"); plt.title("SOC Estimation Comparison")
plt.legend(); plt.tight_layout(); ensure_dir(Path('results'))
plt.savefig('results/soc_comparison.png', dpi=300); plt.show()

print("--- SOH 估计 ---")
soh = soh_capacity_fade(cycle, cycle.soc_true)
print(f"容量衰减法估计 SOH ≈ {soh*100:.2f} %")

np.savez('results/soc_estimates.npz',
         soc_true=cycle.soc_true,
         soc_cc=soc_cc,
         soc_adj=soc_adj,
         soc_ekf=soc_ekf,
         soc_ukf=soc_ukf,
         time=cycle.t)
print("数据已保存至 results/soc_estimates.npz")
