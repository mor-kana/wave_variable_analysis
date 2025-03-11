import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

#figure()でグラフを表示する領域をつくり，figというオブジェクトにする．
fig = plt.figure()

#add_subplot()でグラフを描画する領域を追加する．引数は行，列，場所
ax1 = fig.add_subplot(4, 2, 1)
ax2 = fig.add_subplot(4, 2, 2)
ax3 = fig.add_subplot(4, 2, 3)
ax4 = fig.add_subplot(4, 2, 4)
ax5 = fig.add_subplot(4, 2, 5)
ax6 = fig.add_subplot(4, 2, 6)
ax7 = fig.add_subplot(4, 2, 7)
ax8 = fig.add_subplot(4, 2, 8)

t = np.linspace(-2, 10, 13)
y1 = np.sin(t)
y2 = np.cos(t) 
y3 = np.abs(np.sin(t))
y4 = np.sin(t)**2
y5 = np.sin(t)
y6 = np.cos(t) 
y7 = np.abs(np.sin(t))
y8 = np.sin(t)**2

c1,c2,c3,c4,c5,c6,c7,c8 = "black","black","black","black","black","black","black","black"# 各プロットの色
l1,l2,l3,l4,l5,l6,l7,l8 = "sin","cos","abs(sin)","sin**2","sin","cos","abs(sin)","sin**2"   # 各ラベル

ax1.plot(t, y1, color=c1, label=l1)
ax2.plot(t, y2, color=c2, label=l2)
ax3.plot(t, y3, color=c3, label=l3)
ax4.plot(t, y4, color=c4, label=l4)
ax5.plot(t, y5, color=c5, label=l5)
ax6.plot(t, y6, color=c6, label=l6)
ax7.plot(t, y7, color=c7, label=l7)
ax8.plot(t, y8, color=c8, label=l8)

ax1.legend(loc = 'upper right') #凡例
ax2.legend(loc = 'upper right') #凡例
ax3.legend(loc = 'upper right') #凡例
ax4.legend(loc = 'upper right') #凡例
ax5.legend(loc = 'upper right') #凡例
ax6.legend(loc = 'upper right') #凡例
ax7.legend(loc = 'upper right') #凡例
ax8.legend(loc = 'upper right') #凡例

fig.tight_layout()              #レイアウトの設定
plt.show()