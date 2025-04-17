import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

plt.rcParams['font.size'] = 10.0
plt.rcParams['font.family'] = "Times New Roman"
plt.rcParams['mathtext.fontset'] = "cm"

# パラメータ設定
num_point = 201  # 100点に増やす
cycle = 1
point_maxtime = 18
t = np.linspace(-2 * cycle, point_maxtime, num_point)  # より多くの点で曲線を滑らかに

operator_input = np.array([[1],[1]]) 
robot_output = np.array([[1],[1]])
b = 1.2
c = 1
# 時間配列の生成
print(t)
# 結果を格納する配列の初期化
alpha_m_value = np.zeros((num_point,2,1), dtype=float)
alpha_s_value = np.zeros((num_point,2,1), dtype=float)
beta_m_value = np.zeros((num_point,2,1), dtype=float)
beta_s_value = np.zeros((num_point,2,1), dtype=float)
um_value = np.zeros((num_point,2,1), dtype=float)
us_value = np.zeros((num_point,2,1), dtype=float)
vm_value = np.zeros((num_point,2,1), dtype=float)
vs_value = np.zeros((num_point,2,1), dtype=float)
#vs_value = np.full_like(num_point,np.sqrt(1/2))

# 基本関数の定義
def r_m(t_value):
    if t_value < 0:
        return np.zeros((2, 1))  # 2x1のゼロ行列
    else:
        return operator_input  # operator_inputを返す

def omega_m(t_value):
    if t_value < 0:
        return np.zeros((2, 1))  # 2x1のゼロ行列
    else:
        return robot_output  # robot_outputを返す


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
        # test
        
        # 対応するインデックスを取得
        idx_minus_2 = get_index_from_time(t, t_minus_2)
        idx_minus_1 = get_index_from_time(t, t_minus_1)

        #print(beta_m_value[i])
        beta_m_value[i] = b * alpha_m_value[i-1] - np.sqrt(2*b) * vm_value[i]

        if np.any(alpha_m_value[idx_minus_2] == 1):
            alpha_m_value[i] = np.zeros((2, 1))
        else:
            alpha_m_value[i] = operator_input
            
        
        vm_value[i] = vs_value[idx_minus_1]
        
        # print(alpha_m_value[i], vm_value[i])
        # um の計算
        um_value[i] = (b * alpha_m_value[i] + beta_m_value[i]) / np.sqrt(2 * b)
        
        # us の計算（時間シフト）
        # if current_time >= 1:
        us_value[i] = um_value[idx_minus_1]
        
        # r_s の計算
        rng = np.random.default_rng()
        alpha_s_value[i] = np.sqrt(2/b) * us_value[i] - beta_s_value[i-1]/b
        beta_s_value[i] = alpha_s_value[i] #* (1 + rng.random() * 0.01)
        print("alpha_s_value")
        print(alpha_s_value[i])
        # alpha_s_value[i] = alpha_m_value[idx_minus_1] + c * (beta_m_value[idx_minus_1]-beta_s_value[i])/b
        
        # vs の計算
        vs_value[i] = (b * alpha_s_value[i] - beta_s_value[i]) / np.sqrt(2 * b)
        

#########################################################################< グラフ設定 >#####################################################################################
r_m_v = np.zeros(num_point, dtype=float)
r_s_v = np.zeros(num_point, dtype=float)
beta_m_p = np.zeros(num_point, dtype=float)
beta_s_p = np.zeros(num_point, dtype=float)
um_v = np.zeros(num_point, dtype=float)
us_v = np.zeros(num_point, dtype=float)
vm_p = np.zeros(num_point, dtype=float)
vs_p = np.zeros(num_point, dtype=float)

r_m_omega = np.zeros(num_point, dtype=float)
r_s_omega = np.zeros(num_point, dtype=float)
beta_m_theta = np.zeros(num_point, dtype=float)
beta_s_theta = np.zeros(num_point, dtype=float)
um_omega = np.zeros(num_point, dtype=float)
us_omega = np.zeros(num_point, dtype=float)
vm_theta = np.zeros(num_point, dtype=float)
vs_theta = np.zeros(num_point, dtype=float)

r_m_v = np.array([x[0, 0] for x in alpha_m_value])
r_s_v = np.array([x[0, 0] for x in alpha_s_value])
beta_m_p = np.array([x[0, 0] for x in beta_m_value])
beta_s_p = np.array([x[0, 0] for x in beta_s_value])
um_v = np.array([x[0, 0] for x in um_value])
us_v = np.array([x[0, 0] for x in us_value])
vm_p = np.array([x[0, 0] for x in vm_value])
vs_p = np.array([x[0, 0] for x in vs_value])

r_m_omega = np.array([x[1, 0] for x in alpha_m_value])
r_s_omega = np.array([x[1, 0] for x in alpha_s_value])
beta_m_theta = np.array([x[1, 0] for x in beta_m_value])
beta_s_theta = np.array([x[1, 0] for x in beta_s_value])
um_omega = np.array([x[1, 0] for x in um_value])
us_omega = np.array([x[1, 0] for x in us_value])
vm_theta = np.array([x[1, 0] for x in vm_value])
vs_theta = np.array([x[1, 0] for x in vs_value])

#好みのグリッドを設定するための関数
def setGridPrefered(ax):
    """
    好みのグリッドを設定するための関数

    Parameters
    ----------
    ax : matplotlibの軸オブジェクト

    """
    
    import matplotlib.ticker as ticker
    ax.grid(which='major', lw=0.7) # 主目盛の描画(標準)
    
    # X,Y軸に対して、(補助目盛)×5 = (主目盛)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax.grid(which='minor', lw=0.4) # 補助目盛の描画

# グラフ領域の作成
fig = plt.figure(figsize=(14, 10))
ax1 = fig.add_subplot(4, 4, 1)
ax2 = fig.add_subplot(4, 4, 2)
ax3 = fig.add_subplot(4, 4, 3)
ax4 = fig.add_subplot(4, 4, 4)
ax5 = fig.add_subplot(4, 4, 5)
ax6 = fig.add_subplot(4, 4, 6)
ax7 = fig.add_subplot(4, 4, 7)
ax8 = fig.add_subplot(4, 4, 8)
ax9 = fig.add_subplot(4, 4, 9)
ax10= fig.add_subplot(4, 4, 10)
ax11= fig.add_subplot(4, 4, 11)
ax12= fig.add_subplot(4, 4, 12)
ax13= fig.add_subplot(4, 4, 13)
ax14= fig.add_subplot(4, 4, 14)
ax15= fig.add_subplot(4, 4, 15)
ax16= fig.add_subplot(4, 4, 16)


# プロット設定
c1,c2,c3,c4,c5,c6,c7,c8 = "black","black","black","black","black","black","black","black"
c9,c10,c11,c12,c13,c14,c15,c16 = "black","black","black","black","black","black","black","black"
l1, l2, l3, l4 = r"$\alpha_{mv}$ [m/s]", r"$u_{mv}$ [m/s]", r"$u_{sv}$ [m/s]", r"$\alpha_{sv}$ [m/s]"
l5, l6, l7, l8 = r"$\beta_{sp}$ [m", r"$v_{sp}$ [m]", r"$v_{mp}$ [m]", r"$\beta_{mp}$ [m]"
l9, l10, l11, l12 = r"$\alpha_{m\omega}$ [rad/s]", r"$u_{m\omega}$ [rad/s]", r"$u_{s\omega}$ [rad/s]", r"$\alpha_{s\omega}$ [rad/s]"
l13, l14, l15, l16 = r"$\beta_{s\theta}$ [rad]", r"$v_{s\theta}$ [rad]", r"$v_{m\theta}$ [rad]", r"$\beta_{m\theta}$ [rad]"

# 関数のプロット
ax1.plot(t, r_m_v, color=c1, label=l1, marker='.')
ax2.plot(t, beta_m_p, color=c8, label=l8, marker='.')
ax5.plot(t, um_v, color=c2, label=l2, marker='.')
ax6.plot(t, vm_p, color=c7, label=l7, marker='.')
ax9.plot(t, us_v, color=c3, label=l3, marker='.')
ax10.plot(t, vs_p, color=c6, label=l6, marker='.')
ax13.plot(t, r_s_v, color=c4, label=l4, marker='.')
ax14.plot(t, beta_s_p, color=c5, label=l5, marker='.')

ax3.plot(t, r_m_omega, color=c1, label=l9, marker='.')
ax4.plot(t, beta_m_theta, color=c8, label=l16, marker='.')
ax7.plot(t, um_omega, color=c2, label=l10, marker='.')
ax8.plot(t, vm_theta, color=c7, label=l15, marker='.')
ax11.plot(t, us_omega, color=c3, label=l11, marker='.')
ax12.plot(t, vs_theta, color=c6, label=l14, marker='.')
ax15.plot(t, r_s_omega, color=c4, label=l12, marker='.')
ax16.plot(t, beta_s_theta, color=c5, label=l13, marker='.')

ax_list = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14, ax15, ax16]
label_list = [l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15, l16]
# 凡例,値の表示範囲の追加
for i, ax in enumerate(ax_list):
   # ax.legend(loc='upper right')
    ax.set_ylim(-3, 3)
    ax.set_ylabel(label_list[i])
    ax.set_xlabel('Time [s]')
    setGridPrefered(ax)

# レイアウト設定とプロット表示
fig.tight_layout()
plt.subplots_adjust(wspace=0.25, left=0.05, right=0.9)#余白設定 

for ax in [ax3, ax4, ax7, ax8, ax11, ax12, ax15, ax16]:
    pos = ax.get_position()
    # 右側のサブプロットを右に0.05（図の幅の5%）移動
    ax.set_position([pos.x0 + 0.05, pos.y0, pos.width, pos.height])

plt.draw() 

plt.draw()  # レイアウトを計算するために一度描画

# 中央のx座標を計算
bbox1 = ax2.get_position()  # 左側の最も右にあるサブプロット
bbox2 = ax3.get_position()  # 右側の最も左にあるサブプロット
center_x = (bbox1.x1 + bbox2.x0) / 2

# 縦線を引く（上から下まで）
line = plt.Line2D([center_x, center_x], [0, 1], transform=fig.transFigure, color='black', linewidth=2)
fig.add_artist(line)

# # グラフ上部に「左側グラフ」と「右側グラフ」のラベルを追加
# fig.text(bbox1.x0 + (bbox1.x1 - bbox1.x0) / 2, 0.98, '左側グラフ群', ha='center', va='top', fontsize=12)
# fig.text(bbox2.x0 + (bbox2.x1 - bbox2.x0) / 2, 0.98, '右側グラフ群', ha='center', va='top', fontsize=12)


plt.show()