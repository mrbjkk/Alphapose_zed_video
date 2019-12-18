import readkpt as rpt
import ipdb
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

readpath = 'testdata/running/kpt8.txt'
alldata, lknee, lankle = rpt.readkpt(readpath, 11)
lknee = lknee.reshape((int(len(lknee)/3), 3))
lankle = lankle.reshape((int(len(lankle)/3), 3))
lkneex = []
lkneey = []
lkneez = []
lanklex = []
lankley = []
lanklez = []

# fig1 = plt.figure()
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

# calculate the angle of knee-ankle vector and x axis
veck2a = []
veck2a = np.array(veck2a)
xaxis = np.array([1,0])
yaxis = np.array([0,-1])
cos_anglex = []
cos_angley = []
for i in range(len(lkneex)):
    veck2a = np.append(veck2a, (lanklex[i]-lkneex[i], lankley[i]-lkneey[i]))
veck2a = veck2a.reshape((int(len(veck2a)/2), 2))

for i in range(len(veck2a)):
    cos_anglex = np.append(cos_anglex, veck2a[i].dot(xaxis) /
                          (np.sqrt(veck2a[i].dot(veck2a[i])) * np.sqrt(xaxis.dot(xaxis))))
    cos_angley = np.append(cos_angley, veck2a[i].dot(yaxis) /
                          (np.sqrt(veck2a[i].dot(veck2a[i])) * np.sqrt(yaxis.dot(yaxis))))

radianx = np.arccos(cos_anglex)
anglex = radianx * 360 / 2 / np.pi
radiany = np.arccos(cos_angley)
angley = radiany * 360 / 2 / np.pi
ipdb.set_trace()
fig1 = plt.figure()
for i in range(len(anglex[:200])):
    plt.bar(i, anglex[i])
plt.show()
ipdb.set_trace()
# calculate lknee & lankle vector projected to x,y plain
projdist = []
projdist = np.array(projdist)
# for i, j in zip(lkneex, lanklex):
#     for k, l in zip(lkneey, lankley):
#         projdist = np.append(projdist, np.sqrt((i-j)**2 + (k-l)**2))
for i in range(len(lanklex)):
    projdist = np.append(projdist, np.sqrt((lkneex[i]-lanklex[i])**2 + (lkneey[i]-lankley[i])**2))

fig1 = plt.figure()
for i in range(len(projdist[:200])):
    plt.bar(i, projdist[i])
plt.show()
ipdb.set_trace()
