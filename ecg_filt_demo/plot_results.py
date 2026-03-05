import numpy as np
import matplotlib.pyplot as plt
import sys

fs = 1000

data = np.loadtxt("ecg_filtered.dat")

filtered = data[:,0]
orig = data[:,1]
remover = data[:,2]
w1 = data[:,3]
w2 = data[:,4]

t = np.linspace(0,1/fs*len(orig),len(orig))

fig=plt.figure()
ax1 = plt.subplot(411)
ax2 = plt.subplot(412, sharex = ax1)
ax3 = plt.subplot(413, sharex = ax1)
ax4 = plt.subplot(414, sharex = ax1)
ax1.plot(t,orig)
ax1.set_ylabel("Orig/V")
ax2.plot(t,remover)
ax2.set_ylabel("Remover/V")
ax3.plot(t,filtered)
ax3.set_ylabel("Filtered/V")
ax4.plot(t,w1)
ax4.plot(t,w2)
ax4.set_ylabel("|w1|,|w2|")
plt.xlabel("time/sec")
plt.show()
