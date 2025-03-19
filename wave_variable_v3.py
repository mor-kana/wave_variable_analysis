import numpy as np
import matplotlib.pyplot as plt

# パラメータ設定
num_point = 61  # 100点に増やす
cycle = 1
point_maxtime = 10
operator_input = 1
robot_output = 0
b = 1
c = 1
# 時間配列の生成
t = np.linspace(-2 * cycle, point_maxtime, num_point)  # より多くの点で曲線を滑らかに
print(t)
# 結果を格納する配列の初期化
delta_m_value = np.zeros_like(t, dtype=float)
delta_s_value = np.zeros_like(t, dtype=float)
omega_m_value = np.zeros_like(t, dtype=float)
omega_s_value = np.zeros_like(t, dtype=float)
um_value = np.zeros_like(t, dtype=float)
us_value = np.zeros_like(t, dtype=float)
vm_value = np.zeros_like(t, dtype=float)
vs_value = np.zeros_like(t, dtype=float)

# 基本関数の定義
def delta_m(t_value):
    return 0 if t_value < 0 else operator_input

def omega_s(t_value):
    return 0 if t_value < 0 else robot_output

# まず基本入出力値を計算
for i in range(num_point):
    delta_m_value[i] = delta_m(t[i])

# t値からインデックスを取得する関数
def get_index_from_time(t_array, t_value):
    # t_valueに最も近い時間のインデックスを返す
    return np.argmin(np.abs(t_array - t_value))

# 再帰的な関数の計算
for i in range(num_point):
    current_time = t[i]
    
    # インデックスではなく時間による参照
    if current_time >= 0:
        # t-2と t-1の時間値
        t_minus_2 = current_time - 2
        t_minus_1 = current_time - 1
        
        # 対応するインデックスを取得
        idx_minus_2 = get_index_from_time(t, t_minus_2)
        idx_minus_1 = get_index_from_time(t, t_minus_1)
        
        # print(t[idx_minus_1],t[idx_minus_2])
        
        # omega_m の計算
        # if current_time >= 2:
        #     omega_m_value[i] = (b * delta_m_value[i] - 
        #                        b * delta_m_value[idx_minus_2] - 
        #                        omega_m_value[idx_minus_2] + 
        #                        omega_s_value[idx_minus_1])
        # else:
        #     # 初期条件 (t < 2)
        #     omega_m_value[i] = b * delta_m_value[i] + omega_s_value[idx_minus_1]
        
        # vm の計算（時間シフト）
        # if current_time >= 1:
        vm_value[i] = vs_value[idx_minus_1]
        
        # print(delta_m_value[i], vm_value[i])
        omega_m_value[i] = b * delta_m_value[i] - np.sqrt(2*b) * vm_value[i]
        # um の計算
        um_value[i] = (b * delta_m_value[i] + omega_m_value[i]) / np.sqrt(2 * b)
        
        # us の計算（時間シフト）
        # if current_time >= 1:
        us_value[i] = um_value[idx_minus_1]
        
        # delta_s の計算
        rng = np.random.default_rng()
        omega_s_value[i] = delta_s_value[i-1] * (1 + rng.random() * 0.01)
        delta_s_value[i] = np.sqrt(2/b) * us_value[i] - omega_s_value[i]/b
        # print(delta_s_value[i])
        # delta_s_value[i] = delta_m_value[idx_minus_1] + c * (omega_m_value[idx_minus_1]-omega_s_value[i])/b
        
        # vs の計算
        vs_value[i] = (b * delta_s_value[i] - omega_s_value[i]) / np.sqrt(2 * b)
        

# グラフ領域の作成
fig = plt.figure(figsize=(14, 10))
ax1 = fig.add_subplot(4, 2, 1)
ax2 = fig.add_subplot(4, 2, 2)
ax3 = fig.add_subplot(4, 2, 3)
ax4 = fig.add_subplot(4, 2, 4)
ax5 = fig.add_subplot(4, 2, 5)
ax6 = fig.add_subplot(4, 2, 6)
ax7 = fig.add_subplot(4, 2, 7)
ax8 = fig.add_subplot(4, 2, 8)

# プロット設定
c1,c2,c3,c4,c5,c6,c7,c8 = "black","black","black","black","black","black","black","black"
l1,l2,l3,l4,l5,l6,l7,l8 = "delta_m","um","us","delta_s","omega_s","vs","vm","omega_m"

# 関数のプロット
ax1.plot(t, delta_m_value, color=c1, label=l1, marker='.')
ax2.plot(t, omega_m_value, color=c8, label=l8, marker='.')
ax3.plot(t, um_value, color=c2, label=l2, marker='.')
ax4.plot(t, vm_value, color=c7, label=l7, marker='.')
ax5.plot(t, us_value, color=c3, label=l3, marker='.')
ax6.plot(t, vs_value, color=c6, label=l6, marker='.')
ax7.plot(t, delta_s_value, color=c4, label=l4, marker='.')
ax8.plot(t, omega_s_value, color=c5, label=l5, marker='.')

# 凡例,値の表示範囲の追加
for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:
    ax.legend(loc='upper right')
    ax.set_ylim(-3, 3)

# レイアウト設定とプロット表示
fig.tight_layout()
plt.show()