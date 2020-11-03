import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2]])


def xyz_rot(pos, ori):
    rot = qvec2rotmat(ori)
    rot_t = np.transpose(rot)
    rot_neg = -1 * rot_t

    xyz_ = np.dot(rot_neg, pos)
    angle_ = np.dot(rot, [1.0, 1.0, 1.0])
    return xyz_, angle_


fig = plt.figure()

bottom, top = plt.xlim()
plt.ylim((-10, 10))
#
# file_object = open("dataset_train.txt", "r")
file_object = open("plot_exp_house.txt", "r")

# rot = open("rot.txt", "r")
list_from_text_file = file_object.read().split()

# print((list_from_text_file))

n = (int(len(list_from_text_file) / 8))

pos = []
quat = []
xyz_ = np.zeros((n, 3))
angle_ = np.zeros((n, 3))

for i in range(0, int(len(list_from_text_file)), 8):
    pos.append(float(list_from_text_file[i + 1]))
    pos.append(float(list_from_text_file[i + 2]))
    pos.append(float(list_from_text_file[i + 3]))

    quat.append(float(list_from_text_file[i + 4]))
    quat.append((float(list_from_text_file[i + 5])))
    quat.append((float(list_from_text_file[i + 6])))
    quat.append((float(list_from_text_file[i + 7])))

quat = np.asarray(quat).reshape(n, 4)
pos = np.asarray(pos).reshape(n, 3)

for i in range(0, n):
    xyz_[i], angle_[i] = xyz_rot(pos[i], quat[i])

#
pca2 = PCA(n_components=2)
Reduced_pose = np.zeros((n, 2))
Reduced_rot = np.zeros((n, 2))

#
Reduced_pose[:, 0:2] = pca2.fit_transform(xyz_)
Reduced_rot[:, 0:2] = pca2.fit_transform(angle_)

#

frequency = 5
plt.quiver(Reduced_pose[::frequency, 0], Reduced_pose[::frequency, 1], Reduced_rot[::frequency, 0],
           Reduced_rot[::frequency, 1], width=0.001, color="blue")

plt.show()
