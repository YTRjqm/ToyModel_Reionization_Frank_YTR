import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftn, ifftn, fftfreq
import os

# ========== 参数 ==========
box_size = 50.0  # Mpc/h
N = 512          # 网格数
zeta = 1.0       # 电离效率参数
rho_mean = 1.0   # 平均质量密度（单位化）
delta_c = 5e-5   # 高密度阈值

output_dir = "C:/Users/YTR/Desktop/FinalYTR"
os.makedirs(output_dir, exist_ok=True)

# ========== Step 1: 加载/生成初始密度场 ==========
try:
    delta_x = np.load("C:/Users/YTR/Desktop/FinalYTR/linear_density_field_z100.npy")
    print("Loaded delta_x from file.")
except FileNotFoundError:
    print("No delta_x file found, generating Gaussian random field...")
    kx = fftfreq(N, d=box_size/N) * 2 * np.pi
    ky, kz = kx.copy(), kx.copy()
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    k_mag = np.sqrt(KX**2 + KY**2 + KZ**2)
    k_mag[0, 0, 0] = 1e-10
    P_k = k_mag**-3
    P_k[k_mag < 1e-2] = 0.0
    noise = np.random.normal(size=(N, N, N)) + 1j * np.random.normal(size=(N, N, N))
    delta_k = noise * np.sqrt(P_k)
    delta_x = np.real(ifftn(delta_k))
    delta_x -= np.mean(delta_x)
    np.save("linear_density_field_z100.npy", delta_x)

# ========== Step 2: 光子产额场 ==========
rho_field = rho_mean * (1 + delta_x)
photon_yield_field = 1 * zeta * rho_field * (delta_x > delta_c)

# ========== Step 3: 绘制切片 ==========
slice_index = N // 2  # 中间切片
photon_slice = photon_yield_field[:, :, slice_index]

plt.figure(figsize=(6, 5))
plt.imshow(photon_slice, origin='lower', cmap='inferno',
           extent=[0, box_size, 0, box_size])
plt.xlabel('Mpc/h')
plt.ylabel('Mpc/h')
plt.title('Photon Yield Slice at z = midplane')
plt.colorbar(label='Photon Yield')
plt.tight_layout()
plt.savefig(f"{output_dir}/photon_yield_slice.png")
plt.close()

print("Photon yield slice saved.")
