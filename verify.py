# verification
import readkpt as rpt
import pdb
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

readpath = 'testdata/xyverif/rkpt.txt'
savepath = 'testdata/xyverif/bar.jpg'
interval = 5
truexvalue = 600
alldata, lankle = rpt.readkpt(readpath)
lankle = lankle.reshape((int(len(lankle)/3), 3))
lanklex = []
lankley = []
lanklez = []

fig1 = plt.figure()
# fig2 = plt.figure()
# ax = Axes3D(fig2)

for i in range(int(len(lankle))):
    if lankle[i,0] < 1000.0 and lankle[i,0] > -1000.0:
        lanklex = np.append(lanklex, lankle[i,0])
    if lankle[i,1] < 1000.0 and lankle[i,1] > 0:
        lankley = np.append(lankley, lankle[i,1])
    if lankle[i,2] < 4000.0 and lankle[i,2] > 0:
        lanklez = np.append(lanklez, lankle[i,2])

for i in range(len(lanklex)):
    plt.bar(i, lanklex[i])

x = np.arange(0, len(lanklex))
y = np.arange(0, len(lankley))
z = np.arange(0, len(lanklez))
f = np.polyfit(x, lanklex, 3)
p = np.poly1d(f)

plt.plot(x, lanklex)
plt.show()
'''
RMSE calculation for x,y,z
'''
_xsum = []
_ysum = []
_zsum = []
xrmselist = []
for i in range(0, len(lanklex[:35]), interval):
    for j in lanklex[i:i+interval]:
        _xsum.append((j-truexvalue)**2)
    xrmselist.append(np.sqrt(np.sum(_xsum[i:i+interval])/interval))
meanxrmse = np.mean(xrmselist)
pdb.set_trace()

# plt.savefig(savepath)

plt.show()
