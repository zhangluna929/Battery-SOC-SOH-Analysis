"""
SOC / SOH 估计算法示例
包含：
1. 库伦计数 + 开路电压修正
2. 一阶扩展卡尔曼滤波（EKF）
"""

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. 模拟工况数据
# -----------------------------
np.random.seed(0)
N  = 500          # 时间步
dt = 1            # s
cap_nominal = 2.5 # Ah

# 电流工况：放电 -5 A → 充电 3 A → 放电 -2 A
current = np.concatenate([
    np.full(100, -5),
    np.full(100,  3),
    np.full(300, -2)
]) + np.random.normal(0, 0.1, N)

# 假设 OCV-SOC 多项式模型
poly_ocv = np.poly1d([-0.1, 0.3, 3.5])

# 真值 SOC 曲线
true_soc = 0.8 + np.cumsum(-current * dt / 3600) / cap_nominal
true_soc = np.clip(true_soc, 0, 1)
voltage  = poly_ocv(true_soc) + np.random.normal(0, 0.01, N)

# -----------------------------
# 2. 库伦计数
# -----------------------------
soc_cc = np.zeros(N)
soc_cc[0] = 0.8
for k in range(1, N):
    soc_cc[k] = soc_cc[k-1] - current[k] * dt / (3600 * cap_nominal)
    soc_cc[k] = np.clip(soc_cc[k], 0, 1)

# -----------------------------
# 3. EKF
# -----------------------------
soc_ekf = np.zeros(N)
soc_ekf[0] = 0.8
P, Q, R = 1e-3, 1e-5, 1e-4

for k in range(1, N):
    # 预测
    soc_pred = soc_ekf[k-1] - current[k] * dt / (3600 * cap_nominal)
    soc_pred = np.clip(soc_pred, 0, 1)
    P_pred   = P + Q

    # 观测模型线性化
    H = np.polyder(poly_ocv)(soc_pred)
    K = P_pred * H / (H * P_pred * H + R)  # 卡尔曼增益

    # 更新
    soc_ekf[k] = soc_pred + K * (voltage[k] - poly_ocv(soc_pred))
    P = (1 - K * H) * P_pred

# -----------------------------
# 4. 绘图与 SOH 简估
# -----------------------------
plt.figure(figsize=(6, 4))
plt.plot(true_soc, label="True SOC")
plt.plot(soc_cc,   "--", label="Coulomb Count")
plt.plot(soc_ekf,  ":",  label="EKF")
plt.xlabel("Time step"); plt.ylabel("SOC")
plt.title("SOC Estimation")
plt.legend(); plt.tight_layout()
plt.savefig("soc_estimation.png", dpi=300)
plt.show()

# 以累计放电量估算 SOH（示例）
soh_est = (np.sum(-current[current < 0]) * dt / 3600) / cap_nominal
print(f"估计 SOH ≈ {soh_est * 100:.1f} %")
