# verification
import readkpt as rpt
import ipdb
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

# readpath = 'testdata/xyverif/rkpt.txt'
# savepath = 'testdata/xyverif/bar.jpg'
# readpath = 'testdata/20191014/walkstraightly/kpt.txt'
# savepath = 'testdata/20191014/walkstraightly/bar.jpg'
readpath = 'testdata/running/kpt8.txt'
alldata, lknee, lankle = rpt.readkpt(readpath, 11)
ipdb.set_trace()
lknee = lknee.reshape((int(len(lknee)/3), 3))
lankle = lankle.reshape((int(len(lankle)/3), 3))
lkneex = []
lkneey = []
lkneez = []
lanklex = []
lankley = []
lanklez = []

fig1 = plt.figure()
# fig2 = plt.figure()
# ax = Axes3D(fig2)

# fill the lanklex,y,z
for i in range(int(len(lknee))):
#     if lankle[i,0] < 1000.0 and lankle[i,0] > -1000.0:
        lkneex = np.append(lkneex, lknee[i,0])
#     if lankle[i,1] < 1000.0 and lankle[i,1] > 0:
        lkneey = np.append(lkneey, lknee[i,1])
#     if lankle[i,2] < 30000.0 and lankle[i,2] > 0:
        lkneez = np.append(lkneez, lknee[i,2])
for i in range(int(len(lankle))):
#     if lankle[i,0] < 0 and lankle[i,0] > -800.0:
        lanklex = np.append(lanklex, lankle[i,0])
#     if lankle[i,1] < 1000.0 and lankle[i,1] > 0:
        lankley = np.append(lankley, lankle[i,1])
#     if lankle[i,2] < 30000.0 and lankle[i,2] > 0:
        lanklez = np.append(lanklez, lankle[i,2])


for i in range(len(lkneey[:100])):
    plt.bar(i, lkneey[i])

x = np.arange(0, len(lkneez[:100]))
y = np.arange(0, len(lkneey[:100]))
z = np.arange(0, len(lkneez[:100]))
f = np.polyfit(y, lkneey[:100], 3)
p = np.poly1d(f)

plt.plot(y, lkneey[:100])
plt.show()
ipdb.set_trace()
'''
RMSE calculation for x,y,z
'''
# mean rmse / interval = 5
interval = 3
truexvalue = 600
_xsum = []
_ysum = []
_zsum = []
xrmselist = []
for i in range(0, len(lanklex[:36]), interval):
    for j in lanklex[i:i+interval]:
        _xsum.append((j-truexvalue)**2)
    xrmselist.append(np.sqrt(np.sum(_xsum[i:i+interval])/interval))
meanxrmse = np.mean(xrmselist)

# rmse
xsum = 0
for i in lanklex[:35]:
    xsum += (i-truexvalue)**2
xrmse = np.sqrt(xsum/len(lanklex[:35]))
ipdb.set_trace()

# plt.savefig(savepath)

plt.show()
