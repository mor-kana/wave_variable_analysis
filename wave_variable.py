import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt


operator_input = 1
robot_output = 0
b = 1

# 配列入力に対応するようベクトル化した関数
def delta_m(t):
    return np.where(t < 0, 0, operator_input)

def omega_s(t):
    return np.where(t < 0, 0, robot_output)

# omega_mは再帰的かつ複雑なので、別の方法で処理
def omega_m(t_array):
    result = np.zeros_like(t_array, dtype=float)
    
    # 各時点を個別に処理
    for i, t in enumerate(t_array):
        if t < 0:
            result[i] = 0
        elif t >= 0 and t < 2:
            # t-2が負になる場合
            result[i] = b * delta_m(t) + omega_s(t-1)
        else:  # t >= 2
            # 既に計算された配列から前の値を見つける
            # 離散点で作業しているため、これは近似値
            idx_t_minus_2 = np.argmin(np.abs(t_array - (t-2)))
            result[i] = b * delta_m(t) - b * delta_m(t-2) - result[idx_t_minus_2] + omega_s(t-1)
    
    return result

def um(t):
    return (b * delta_m(t) + omega_m(t))/np.sqrt(2 * b)

def us(t):
    # 配列入力の場合、すべての要素をシフト
    result = np.zeros_like(t)
    mask = t >= 1  # t-1 >= 0となる要素のみ
    result[mask] = um(t[mask]-1)
    return result

def delta_s(t):
    return np.sqrt(2/b) * us(t) - omega_s(t)/b

def vs(t):
    return (b * delta_s(t) - omega_s(t))/np.sqrt(2 * b)

def vm(t):
    # usと同様に1だけシフト
    result = np.zeros_like(t)
    mask = t >= 1
    result[mask] = vs(t[mask]-1)
    return result


# 時間点の生成
t = np.linspace(-2, 10, 100)  # より滑らかな曲線のためにポイント数を増加

# 他の関数で使用されるためにomega_mをすべての時間点に対して先に計算
omega_m_values = omega_m(t)

# グラフ領域の作成
fig = plt.figure()
ax1 = fig.add_subplot(4, 2, 1)
ax2 = fig.add_subplot(4, 2, 2)
ax3 = fig.add_subplot(4, 2, 3)
ax4 = fig.add_subplot(4, 2, 4)
ax5 = fig.add_subplot(4, 2, 5)
ax6 = fig.add_subplot(4, 2, 6)
ax7 = fig.add_subplot(4, 2, 7)
ax8 = fig.add_subplot(4, 2, 8)

# プロット設定
c1,c2,c3,c4,c5,c6,c7,c8 = "black","black","black","black","black","black","black","black"  # 色
l1,l2,l3,l4,l5,l6,l7,l8 = "delta_m","um","us","delta_s","omega_s","vs","vm","omega_m"  # ラベル


# 関数のプロット
ax1.plot(t, delta_m(t), color=c1, label=l1)
ax3.plot(t, um(t), color=c2, label=l2)
ax5.plot(t, us(t), color=c3, label=l3)
ax7.plot(t, delta_s(t), color=c4, label=l4)
ax8.plot(t, omega_s(t), color=c5, label=l5)
ax6.plot(t, vs(t), color=c6, label=l6)
ax4.plot(t, vm(t), color=c7, label=l7)
ax2.plot(t, omega_m_values, color=c8, label=l8)

# 凡例の追加
for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:
    ax.legend(loc='upper right')

# レイアウト設定とプロット表示
fig.tight_layout()
plt.show()