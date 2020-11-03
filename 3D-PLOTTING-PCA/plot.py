import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

fig = plt.figure()

bottom, top = plt.xlim()
plt.ylim((-4, 4))

file_object = open("7DOF_to_3D.txt", "r")
rot = open("rot.txt", "r")
list_from_text_file = file_object.read().split()
param_from_file = rot.read().split()

X_ = []
Y_ = []
Z_ = []
X_rot = []
Y_rot = []
Z_rot = []

for i in range(0, int(len(list_from_text_file)), 3):
    X_.append((float(list_from_text_file[i])))
    Y_.append((float(list_from_text_file[i + 1])))
    Z_.append((float(list_from_text_file[i + 2])))

for i in range(0, int(len(param_from_file)), 3):
    X_rot.append((float(param_from_file[i])))
    Y_rot.append((float(param_from_file[i + 1])))
    Z_rot.append((float(param_from_file[i + 2])))

cord = np.zeros([int(len(list_from_text_file) / 3), 3])
param = np.zeros([int(len(param_from_file) / 3), 3])

cord[:, 0] = np.array(X_)
cord[:, 1] = np.array(Y_)
cord[:, 2] = np.array(Z_)

param[:, 0] = np.array(X_rot)
param[:, 1] = np.array(Y_rot)
param[:, 2] = np.array(Z_rot)

pca2 = PCA(n_components=2)
Reduced_pose = np.zeros([int(len(list_from_text_file) / 3), 2])
Reduced_rot = np.zeros([int(len(param_from_file) / 3), 2])

Reduced_pose[:, 0:2] = pca2.fit_transform(cord)
Reduced_rot[:, 0:2] = pca2.fit_transform(param)

frequency = 6
plt.quiver(Reduced_pose[::frequency, 0], Reduced_pose[::frequency, 1], Reduced_rot[::frequency, 0],
           Reduced_rot[::frequency, 1], width=0.001, color="blue")
plt.show()
