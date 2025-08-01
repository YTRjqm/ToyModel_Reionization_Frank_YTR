import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftn, ifftn, fftfreq
from tqdm import tqdm
import os

# ========== 参数 ==========
box_size = 50.0  # Mpc/h
N = 512          # 网格数
zeta = 1.0       # 电离效率参数
rho_mean = 1.0   # 平均质量密度（单位化）
delta_c = 5e-5   # 高密度阈值
n_dirs = 100     # 每个像素发射的射线数

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

# ========== Step 3: 傅里叶平滑判断电离 ==========
kx = fftfreq(N, d=box_size / N) * 2 * np.pi
ky = kx.copy()
kz = kx.copy()
KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
k_mag = np.sqrt(KX**2 + KY**2 + KZ**2)
k_mag[0, 0, 0] = 1e-10
rho_k = fftn(rho_field)
photon_k = fftn(photon_yield_field)

radii = np.logspace(np.log10(1), np.log10(box_size / 2), num=10)  # Mpc/h
ionized_mask = np.zeros((N, N, N), dtype=bool)

def spherical_tophat(k, R):
    x = k * R
    W = 3 * (np.sin(x) - x * np.cos(x)) / (x**3)
    W[x == 0] = 1.0
    return W

# --- 定义方向向量：8 个方向 ---
directions = []
for dx in [-1, 0, 1]:
    for dy in [-1, 0, 1]:
        if dx == 0 and dy == 0:
            continue
        vec = np.array([dx, dy], dtype=float)
        vec /= np.linalg.norm(vec)  # 单位化
        directions.append(vec)

for i, R in enumerate(tqdm(radii, desc="Smoothing & Output")):
    W_k = spherical_tophat(k_mag, R)
    smoothed_photon = np.real(ifftn(photon_k * W_k))
    smoothed_mass = np.real(ifftn(rho_k * W_k))
    ionized_mask_step = smoothed_photon >= smoothed_mass
    ionized_mask |= ionized_mask_step  # 累加式电离

    # ---------- 2D电离切片 ----------
    plt.figure(figsize=(6, 5))
    plt.imshow(ionized_mask[:, :, N // 2], origin='lower', cmap='gray',
               extent=[0, box_size, 0, box_size])
    plt.xlabel('Mpc/h')
    plt.ylabel('Mpc/h')
    plt.title(f'Ionized Slice @ R = {R:.2f} Mpc/h')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/slice_{i:02d}_R{R:.2f}.png")
    plt.close()

    # ---------- 当前体积分数 ----------
    xi_v_step = np.mean(ionized_mask)
    print(f"Step {i+1}/{len(radii)} @ R={R:.2f} Mpc/h → Ionized fraction = {xi_v_step:.3f}")

    # ---------- 当前 MFP 泡泡测量 ----------
    slice_z = N // 2
    slice_mask = ionized_mask[:, :, slice_z]  # 2D mask
    mfp_map = np.zeros((N, N))

    max_distance = 50.0  # Mpc/h
    max_voxel_steps = max_distance / (box_size / N)  # 最大追踪步数（单位：像素）

    # 遍历切片中所有被电离的像素
    for x in range(N):
        for y in range(N):
            if not slice_mask[x, y]:
                continue  # 只测量电离区域

            distances = []
            for d in directions:
                pos = np.array([x + 0.5, y + 0.5])  # 当前点中心位置
                step = d * 0.5
                step_count = 0
                while True:
                    pos += step
                    step_count += 1
                    if step_count > max_voxel_steps:
                        break  # 超过最大距离
                    grid = np.floor(pos).astype(int) % N
                    if not ionized_mask[grid[0], grid[1], slice_z]:
                        break
                dist_voxel = np.linalg.norm(pos - np.array([x + 0.5, y + 0.5]))
                distances.append(dist_voxel * (box_size / N))

            mfp_map[x, y] = np.mean(distances)

    # ---------- 当前泡泡大小直方图 ----------
    bubble_sizes = mfp_map[mfp_map > 0]
    if len(bubble_sizes) > 0:
        avg_size = np.mean(bubble_sizes)
        print(f"→ ⛱️  Avg bubble size on slice: {avg_size:.2f} Mpc/h")

        for max_range in [20, 25, 35, 50]:
            plt.figure(figsize=(6, 5))
            plt.hist(bubble_sizes, bins=30, color='skyblue', edgecolor='k', range=(0, max_range))
            plt.xlabel("Bubble Size (Mpc/h)")
            plt.ylabel("Pixel Count")
            plt.title(f"Bubble Size Dist. @ R = {R:.2f} Mpc/h (0–{max_range})")
            plt.tight_layout()
            plt.savefig(f"{output_dir}/bubble_hist_R{R:.2f}_range{max_range}.png")
            plt.close()
    else:
        print("→ ⚠️  No ionized pixels detected on slice for bubble size.")

