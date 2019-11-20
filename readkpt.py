import pdb
import numpy as np
import pandas as pd

def readkpt(FilePath):
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

    return alldata, lankle
