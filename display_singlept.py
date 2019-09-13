import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

alldata = []
nose = []
nose = np.array(nose)
lshoulder = []
lshoulder = np.array(lshoulder)
rshoulder = []
rshoulder = np.array(rshoulder)
lwrist = []
lwrist = np.array(lwrist)
rwrist = []
rwrist = np.array(rwrist)
lankle = []
lankle = np.array(lankle)
rankle = []
rankle = np.array(rankle)
Person = []
Person = np.array(Person)

f = open('testdata/original_video/kpt.txt')
sourceInLines = f.readlines()
f.close()

for line in sourceInLines:
    temp1 = line.strip('\n')
    temp2 = temp1.split(',')
    alldata.append(temp2)

alldata = np.array(alldata).astype(np.float)

# print(alldata)
col_idx = np.array([2,3,4])
for i in range(len(alldata)):
    if i % 7 == 0:
        nose = np.append(nose, alldata[[i], col_idx])
    if i % 7 == 1:
        lshoulder = np.append(lshoulder, alldata[[i], col_idx])
    if i % 7 == 2:
        rshoulder = np.append(rshoulder, alldata[[i], col_idx])
    if i % 7 == 3:
        lwrist = np.append(lwrist, alldata[[i], col_idx])
    if i % 7 == 4:
        rwrist = np.append(rwrist, alldata[[i], col_idx])
    if i % 7 == 5:
        lankle = np.append(lankle, alldata[[i], col_idx])
    if i % 7 == 6:
        rankle = np.append(rankle, alldata[[i], col_idx])

lwrist = lwrist.reshape((int(len(lwrist)/3), 3))

fig = plt.figure()
# ax = Axes3D(fig)
count = 0
for i in range(int(len(lwrist))):
#     if lwrist[i,2] > 1500.0 and lwrist[i,2] < 2200.0:
    if lwrist[i,2] < 6000.0 and lwrist[i,2] > 0:
        plt.bar(i, lwrist[i,2])
        count = count + 1

print('reasonable rate is: ', count / len(lwrist))
plt.savefig('testdata/original_video/bar.jpg')

# def animate(i):
#     return ax.scatter(lwrist[i, 0], lwrist[i, 1], lwrist[i, 2])

# anim = animation.FuncAnimation(fig, animate, frames = len(lwrist), interval = 20, blit = False)
plt.show()
