import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

fig = plt.figure()

bottom, top = plt.xlim()
# plt.ylim((-4, 4))

file_object = open("../results.txt", "r")
list_from_text_file = file_object.read().split()
data_points = (int(len(list_from_text_file) / 12))

X_ = []
Y_ = []
Z_ = []
X_rot = []
Y_rot = []
Z_rot = []

X_true = []
Y_true = []
Z_true = []
X_rot_true = []
Y_rot_true = []
Z_rot_true = []

for i in range(0, data_points):
    X_rot.append((float(list_from_text_file[i])))
    Y_rot.append((float(list_from_text_file[i + 1])))
    Z_rot.append((float(list_from_text_file[i + 2])))
    X_rot_true.append((float(list_from_text_file[i + 3])))
    Y_rot_true.append((float(list_from_text_file[i + 4])))
    Z_rot_true.append((float(list_from_text_file[i + 5])))
    X_.append((float(list_from_text_file[i + 6])))
    Y_.append((float(list_from_text_file[i + 7])))
    Z_.append((float(list_from_text_file[i + 8])))
    X_true.append((float(list_from_text_file[i + 9])))
    Y_true.append((float(list_from_text_file[i + 10])))
    Z_true.append((float(list_from_text_file[i + 11])))


pos = np.zeros([data_points, 3])
pos_true = np.zeros([data_points, 3])
rot = np.zeros([data_points, 3])
rot_true = np.zeros([data_points, 3])

pos[:, 0] = np.array(X_)
pos[:, 1] = np.array(Y_)
pos[:, 2] = np.array(Z_)

rot[:, 0] = np.array(X_rot)
rot[:, 1] = np.array(Y_rot)
rot[:, 2] = np.array(Z_rot)

pos_true[:, 0] = np.array(X_true)
pos_true[:, 1] = np.array(Y_true)
pos_true[:, 2] = np.array(Z_true)

rot_true[:, 0] = np.array(X_rot_true)
rot_true[:, 1] = np.array(Y_rot_true)
rot_true[:, 2] = np.array(Z_rot_true)

pca2 = PCA(n_components=2)

pose_2d = np.zeros([data_points, 2])
pose_2d_true = np.zeros([data_points, 2])
rot_2d = np.zeros([data_points, 2])
rot_2d_true = np.zeros([data_points, 2])

pose_2d[:, 0:2] = pca2.fit_transform(pos)
rot_2d[:, 0:2] = pca2.fit_transform(rot)
pose_2d_true[:, 0:2] = pca2.fit_transform(pos_true)
rot_2d_true[:, 0:2] = pca2.fit_transform(rot_true)

frequency = 1
plt.quiver(pose_2d[::frequency, 0], pose_2d[::frequency, 1], rot_2d[::frequency, 0],
           rot_2d[::frequency, 1], width=0.001, color="blue")

plt.quiver(pose_2d_true[::frequency, 0], pose_2d_true[::frequency, 1], rot_2d_true[::frequency, 0],
           rot_2d_true[::frequency, 1], width=0.001, color="red")

plt.savefig('home.svg', dpi = 1200)
plt.show()

