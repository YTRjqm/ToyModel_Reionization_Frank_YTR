import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.fft import fftn, ifftn, fftfreq
from numpy.random import normal

# --- 参数设置 ---
box_size = 50.0  # Mpc/h
N = 512          # 网格数量（每维）
dk = 2 * np.pi / box_size
grid = np.fft.fftfreq(N, d=box_size/N) * 2 * np.pi
kx, ky, kz = np.meshgrid(grid, grid, grid, indexing='ij')
k_mag = np.sqrt(kx**2 + ky**2 + kz**2)
k_mag[0, 0, 0] = 1e-10  # 避免除以 0

# --- 定义近似的ΛCDM功率谱 P(k)（z=100 左右） ---
# 粗略近似：P(k) ∝ kⁿ * T(k)²；这里只取指数型下降（你也可用 CLASS/CAMB 结果替代）
def Pk(k):
    A = 1e4  # 正规化因子（调节功率强度）
    n = 1.0
    k0 = 1  # Mpc/h
    return A * k**n * np.exp(-(k/k0)**2)

P_k = Pk(k_mag)

# --- 随机高斯场 δ_k ---
np.random.seed(42)
Re = normal(size=(N, N, N))
Im = normal(size=(N, N, N))
delta_k = (Re + 1j * Im) * np.sqrt(P_k / 2)

# --- 傅里叶反变换：得到实空间密度扰动 δ(x) ---
delta_x = np.real(ifftn(delta_k))

# --- 可视化中间切片 ---
slice_idx = N // 2
plt.figure(figsize=(7, 6))
plt.imshow(delta_x[:, :, slice_idx], origin='lower', cmap='RdBu_r', extent=[0, box_size, 0, box_size])
plt.colorbar(label=r'$\delta$')
plt.title('Linear Density Contrast Slice (z ~ 100)')
plt.xlabel('Mpc/h')
plt.ylabel('Mpc/h')
plt.tight_layout()
plt.show()

# --- 保存为.npy 文件 ---
np.save("C:/Users/YTR/Desktop/FinalYTR/linear_density_field_z100.npy", delta_x)
print("✅ 线性密度场已保存为 linear_density_field_z100.npy")
