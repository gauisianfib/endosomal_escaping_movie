import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# エンドソームの半径
endosome_radius = 10

# 粒子の数
num_particles = 1000

# 磁性粒子の割合
magnetic_ratio = 0.1  # 全体の10%
num_magnetic_particles = int(num_particles * magnetic_ratio)

# シミュレーションのパラメータ
num_steps = 2000  # 最大ステップ数
step_size = 0.5  # 粒子の移動幅
magnetic_force = 0.1  # 磁石の影響力

# 粒子の初期位置 (エンドソーム内にランダム配置)
np.random.seed(42)
particles = np.random.uniform(-endosome_radius, endosome_radius, (num_particles, 3))

# エンドソームの境界チェック
def check_within_endosome(positions):
    distances = np.linalg.norm(positions, axis=1)
    inside = distances < endosome_radius
    return inside

# 初期位置をエンドソーム内に収める
particles = particles[check_within_endosome(particles)]

# 磁性粒子をランダムに選択
magnetic_indices = np.random.choice(len(particles), num_magnetic_particles, replace=False)
magnetic_mask = np.zeros(len(particles), dtype=bool)
magnetic_mask[magnetic_indices] = True

# 磁石の位置（X,Z平面から少し離れた位置に薄い円柱）
magnet_center = np.array([0, 0, -15])  # 円柱の中心位置
magnet_radius = 8  # 円柱の半径
magnet_height = 1  # 円柱の高さ

# エンドソーム破裂の閾値（磁性粒子の集まり具合による）
rupture_threshold = 0.5  # 磁性粒子の50%が膜上に集まると破裂

# 穴の拡張範囲 (穴の大きさを調整)
hole_radius = 1.0  # 穴の半径
hole_expansion_rate = 0.05  # 穴が広がる速度

# 粒子の運動履歴
positions_history = [particles.copy()]

# 穴の位置と拡張状態
hole_position = None
hole_expansion = False

# シミュレーションステップ
for _ in range(num_steps):
    # すべての粒子がブラウン運動を続ける
    moves = np.random.uniform(-step_size, step_size, particles.shape)
    particles += moves

    # 穴が拡大中の場合、穴の周りから粒子が漏れ出す
    if hole_expansion:
        for i, particle in enumerate(particles):
            if np.linalg.norm(particle - hole_position) < hole_radius:
                particles[i] += np.random.uniform(-step_size, step_size, 3)  # 穴から外に漏れ出す

        hole_radius += hole_expansion_rate  # 穴の拡張
        if hole_radius > endosome_radius:  # 穴がエンドソームの全体を貫通した場合
            hole_expansion = False

    # 磁性粒子に磁石の力を加える
    magnetic_particles = particles[magnetic_mask]
    direction_to_magnet = magnet_center - magnetic_particles
    distances_to_magnet = np.linalg.norm(direction_to_magnet, axis=1, keepdims=True)
    force = magnetic_force * direction_to_magnet / (distances_to_magnet + 1e-5)  # 力の計算
    new_positions = magnetic_particles + force

    # 新しい位置がエンドソーム内に収まっているかチェック
    new_distances = np.linalg.norm(new_positions, axis=1)
    
    # エンドソーム外に出た場合、その位置を元に戻す
    for i in range(len(new_positions)):
        if new_distances[i] >= endosome_radius:
            new_positions[i] = magnetic_particles[i]  # 外に出たら元の位置に戻す

    # エンドソーム内に収めた磁性粒子の新しい位置を更新
    particles[magnetic_mask] = new_positions

    # エンドソーム膜の内側に戻す (跳ね返りモデル)
    for i, particle in enumerate(particles):
        if np.linalg.norm(particle) >= endosome_radius:
            particles[i] -= moves[i]  # 動きを打ち消す（反射モデル）

    # 磁性粒子がエンドソーム膜に集まっているかを確認
    magnetic_positions = particles[magnetic_mask]
    distances_to_surface = np.abs(np.linalg.norm(magnetic_positions, axis=1) - endosome_radius)
    
    # エンドソーム膜に近い位置にある磁性粒子をカウント
    surface_particles = np.sum(distances_to_surface < 0.5)  # 表面から0.5以内に集まった粒子の数
    if surface_particles / num_magnetic_particles > rupture_threshold:  # 磁性粒子の50%が膜に集まった場合
        hole_position = magnetic_positions[np.argmin(distances_to_surface)]  # 最も膜に近い粒子の位置を取得
        hole_expansion = True  # 穴の拡張を開始
        print(f"エンドソームに穴が開きました！穴の位置: {hole_position}")

    # 通常粒子も穴から漏れ出す
    if hole_expansion:
        for i, particle in enumerate(particles[~magnetic_mask]):
            if np.linalg.norm(particle - hole_position) < hole_radius:
                particles[~magnetic_mask][i] += np.random.uniform(-step_size, step_size, 3)  # 穴から外に漏れ出す

    # 位置を履歴に記録
    positions_history.append(particles.copy())

# アニメーション描画
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# エンドソームの境界 (球体) を描画
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = endosome_radius * np.outer(np.cos(u), np.sin(v))
y = endosome_radius * np.outer(np.sin(u), np.sin(v))
z = endosome_radius * np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x, y, z, color='r', alpha=0.2)

# 磁石 (薄い円柱) を描画
phi = np.linspace(0, 2 * np.pi, 100)
z_cylinder = np.linspace(-magnet_height / 2, magnet_height / 2, 2) + magnet_center[2]
x_cylinder = magnet_radius * np.outer(np.cos(phi), np.ones(2)) + magnet_center[0]
y_cylinder = magnet_radius * np.outer(np.sin(phi), np.ones(2)) + magnet_center[1]
ax.plot_surface(x_cylinder, y_cylinder, z_cylinder[np.newaxis, :], color='black', alpha=0.3)

# 粒子を描画
normal_scat = ax.scatter([], [], [], color='b', s=10, label='substance')
magnetic_scat = ax.scatter([], [], [], color='g', s=10, label='MNPs')

# 描画範囲設定
ax.set_xticks([])  # X軸の目盛りを消す
ax.set_yticks([])  # Y軸の目盛りを消す
ax.set_zticks([])  # Z軸の目盛りを消す

ax.set_xlim(-endosome_radius, endosome_radius)
ax.set_ylim(-endosome_radius, endosome_radius)
ax.set_zlim(-20, 5)
ax.set_title("endosomal leakage by MNPs")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()

# 横から見るための視点設定
ax.view_init(elev=10, azim=225)
def update(frame):
    current_positions = positions_history[frame]
    
    # 通常粒子と磁性粒子を分けて更新
    normal_positions = current_positions[~magnetic_mask]
    magnetic_positions = current_positions[magnetic_mask]
    
    normal_scat._offsets3d = (
        normal_positions[:, 0], 
        normal_positions[:, 1], 
        normal_positions[:, 2]
    )
    magnetic_scat._offsets3d = (
        magnetic_positions[:, 0], 
        magnetic_positions[:, 1], 
        magnetic_positions[:, 2]
    )
    return normal_scat, magnetic_scat

ani = FuncAnimation(fig, update, frames=len(positions_history), interval=50, blit=False)

plt.show()
