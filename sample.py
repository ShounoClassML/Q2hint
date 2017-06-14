# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pylab as plt
from scipy.fftpack import fft, ifft

t = np.linspace(0, 1, 1024)  # データ [0, 1] までで 1024 分割
delta = 1.0/1024  # 刻み幅
f0 = 16    # 16Hz
f1 = 128   # 128Hz

x = 0.3 * np.sin(2*f0*np.pi*t) + 0.7 * np.cos(2*f1*np.pi*t)

# t の区間を 4 分割したスペクトログラムを作ってみる
L = 1024 / 4
X = np.zeros((L, 4), dtype='complex')  # スペクトログラムは 周波数としては L までで，区間は８個

X[:,0] = fft( x[0*L:1*L] )
X[:,1] = fft( x[1*L:2*L] )
X[:,2] = fft( x[2*L:3*L] )
X[:,3] = fft( x[3*L:4*L] )

# とりあえず，plot してみる
plt.figure()
plt.subplot(2,1,1)
plt.plot( t, x )

plt.show()

plt.subplot(2,1,2)
# plt.imshow( np.log(np.abs(X)) ) # でも良いけどそれなりに表示を適宜合わせる
# plt.imshow( np.log(np.abs(X[:L/2+1,:])), 
#            aspect = 'auto', interpolation='Nearest' )
plt.imshow( np.log(np.abs(X[:L/2+1,:])), 
           extent = (0.0, 1.0, 1./(2*delta), 0.0),
           aspect = 'auto', interpolation='Nearest' )


