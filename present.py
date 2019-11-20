import readkpt as rpt
import numpy as np
import pandas as pd
import pdb
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

readpath = 'testdata/xyverif/rkpt.txt'
alldata, lankle = rpt.readkpt(readpath)
lankle = lankle.reshape((int(len(lankle)/3), 3))
pdb.set_trace()
lanklez = []
lanklex = []
lankley = []
for i in range(int(len(lankle))):
    if lankle[i,0] < 1000.0 and lankle[i,0] > -1000.0:
        lanklex = np.append(lanklex, lankle[i,0])
    if lankle[i,1] < 1000.0 and lankle[i,1] > 0:
        lankley = np.append(lankley, lankle[i,1])
    if lankle[i,2] < 4000.0 and lankle[i,2] > 0:
        lanklez = np.append(lanklez, lankle[i,2])

fig = plt.figure()
ax = Axes3D(fig)
'''
trajectory of lankle
'''
# ax.set_xlim([0, 4000])
# ax.set_ylim([0, 4000])
# ax.set_zlim([2000, 4000])

def update(i):
    return ax.scatter(lanklex[i], lankley[i], lanklez[i])
ani = animation.FuncAnimation(fig, func = update,frames = 100, interval=100, blit = False)
plt.show()
pdb.set_trace()
# for i in range(len(alldata)):
#     Person = np.append(Person, alldata[[i], col_idx])
# 
# for i in range(len(Person)):
#     if i % 3 == 0:
#         Person[i] = Person[i] / 10
# 
# # print(Person)
# for i in range(len(alldata)):
#     coor_z[[i], 2] = alldata[[i], 4]
# 
# # coor_z = coor_z.reshape(360, 7, 3)
# # print(coor_z)
# Person = Person.reshape(int(len(Person)/21), 7, 3)
# # # print(type(len(Person)/3))
# # 
# stick_defines = [
#     (0, 1),
#     (0, 2),
#     (1, 2),
#     (1, 3),
#     (2, 4),
#     (1, 5),
#     (2, 6)
# ]
# 
# # print(np.shape(Person))
# Person_lines = [ax.plot(Person[0,i,0], Person[0,i,2], Person[0,i,1], 'k-')[0] for i in stick_defines]
# 
# # z_lines = [ax.plot(coor_z[0,i,0], coor_z[0,i,2], coor_z[0,i,1], 'k-')[0] for i in stick_defines]
# def animate(j):
#     for stick_line, i in zip(Person_lines, stick_defines):
#         stick_line._verts3d = Person[j,i,0], Person[j,i,2], Person[j,i,1]
#     return Person_lines
# 
# anim = animation.FuncAnimation(fig, animate, frames = int(len(Person/21)), interval=100, blit=False)
# 
# plt.show()
