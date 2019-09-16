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

f = open('testdata/high_accuracy/kpt.txt')
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

lankle = lankle.reshape((int(len(lankle)/3), 3))
lanklez = []

fig = plt.figure()
# ax = Axes3D(fig)
count = 0

for i in range(int(len(lankle)/3)):
    if lankle[i,2] < 3500.0 and lankle[i,2] > 0:
        lanklez = np.append(lanklez, lankle[i,2])
        count = count + 1

for i in range(len(lanklez)):
#     if lwrist[i,2] < 3000.0 and lwrist[i,2] > 0:
#         lwristz = np.append(lwristz, lwrist[i,2])
    plt.bar(i, lanklez[i])

# for i in range(len(lankle)):
#     if lankle[i,2] < 3500.0 and lankle[i,2] > 0:
#         plt.bar(i, lankle[i,2])

# x = np.arange(0, len(lanklez))
# f = np.polyfit(x, lanklez, 3)
# p = np.poly1d(f)
# print(p)

# plot = plt.plot(x, lanklez)

# # draw expected line
# plt.plot([25, 148], [1800, 3000], c = 'r')
# plt.plot([149, 225], [3000, 1800], c = 'r')
# plt.plot([226,275], [1800, 3000], c = 'r')
# plt.plot([276, 330], [3000, 1800], c = 'r')
print('rate is: ', count / int(len(lankle)/3))
# plt.savefig('testdata/high_accuracy/bar.jpg')

# def animate(i):
#     return ax.scatter(lankle[i, 0], lankle[i, 1], lankle[i, 2])
# 
# anim = animation.FuncAnimation(fig, animate, frames = len(lankle), interval = 20, blit = False)
plt.show()
