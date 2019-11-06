import numpy as np
import pandas as pd
import pdb
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
coor_z = np.zeros((len(alldata), 3))

# print(coor_z)
# print(np.shape(coor_z))

# print(alldata)
# print(np.shape(alldata))
# print(len(alldata))
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

fig = plt.figure()
ax = Axes3D(fig)
# for i in range(0, len(lwrist), 3):
#     ax.scatter(lwrist[i], lwrist[i+1], lwrist[i+2])
# 
# plt.show()
pdb.set_trace()
for i in range(len(alldata)):
    Person = np.append(Person, alldata[[i], col_idx])

for i in range(len(Person)):
    if i % 3 == 0:
        Person[i] = Person[i] / 10

# print(Person)
for i in range(len(alldata)):
    coor_z[[i], 2] = alldata[[i], 4]

# coor_z = coor_z.reshape(360, 7, 3)
# print(coor_z)
Person = Person.reshape(int(len(Person)/21), 7, 3)
# # print(type(len(Person)/3))
# 
stick_defines = [
    (0, 1),
    (0, 2),
    (1, 2),
    (1, 3),
    (2, 4),
    (1, 5),
    (2, 6)
]

# print(np.shape(Person))
Person_lines = [ax.plot(Person[0,i,0], Person[0,i,2], Person[0,i,1], 'k-')[0] for i in stick_defines]

# z_lines = [ax.plot(coor_z[0,i,0], coor_z[0,i,2], coor_z[0,i,1], 'k-')[0] for i in stick_defines]
def animate(j):
    for stick_line, i in zip(Person_lines, stick_defines):
        stick_line._verts3d = Person[j,i,0], Person[j,i,2], Person[j,i,1]
    return Person_lines

# def animate(j):
#     for stick_line, i in zip(z_lines, stick_defines):
#         stick_line._verts3d = z_lines[j,i,0], z_lines[j,i,2], z_lines[j,i,1]
#     return z_lines

anim = animation.FuncAnimation(fig, animate, frames = int(len(Person/21)), interval=100, blit=False)

plt.show()
