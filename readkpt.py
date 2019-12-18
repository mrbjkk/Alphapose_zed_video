import ipdb
import numpy as np
import pandas as pd

def readkpt(FilePath, jointnum):
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
    lhip = []
    lhip = np.array(lhip)
    rhip = []
    rhip = np.array(rhip)
    lknee = []
    lknee = np.array(lknee)
    rknee = []
    rknee = np.array(rknee)
    lankle = []
    lankle = np.array(lankle)
    rankle = []
    rankle = np.array(rankle)
    Person = []
    Person = np.array(Person)

    f = open(FilePath)
    sourceInLines = f.readlines()
    f.close()

    for line in sourceInLines:
        temp1 = line.strip('\n')
        temp2 = temp1.split(',')
        alldata.append(temp2)

    alldata = np.array(alldata).astype(np.float)

    col_idx = np.array([2,3,4])
    for i in range(len(alldata)):
        if i % jointnum == 0:
            nose = np.append(nose, alldata[[i], col_idx])
        if i % jointnum == 1:
            lshoulder = np.append(lshoulder, alldata[[i], col_idx])
        if i % jointnum == 2:
            rshoulder = np.append(rshoulder, alldata[[i], col_idx])
        if i % jointnum == 3:
            lwrist = np.append(lwrist, alldata[[i], col_idx])
        if i % jointnum == 4:
            rwrist = np.append(rwrist, alldata[[i], col_idx])
        if i % jointnum == 5:
            lhip = np.append(lhip, alldata[[i], col_idx])
        if i % jointnum == 6:
            rhip = np.append(rhip, alldata[[i], col_idx])
        if i % jointnum == 7:
            lknee = np.append(lknee, alldata[[i], col_idx])
        if i % jointnum == 8:
            rknee = np.append(rknee, alldata[[i], col_idx])
        if i % jointnum == 9:
            lankle = np.append(lankle, alldata[[i], col_idx])
        if i % jointnum == 10:
            rankle = np.append(rankle, alldata[[i], col_idx])

    return alldata, lknee, lankle
