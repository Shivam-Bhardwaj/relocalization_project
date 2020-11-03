import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# model = 'KingsCollege/loss'

pose_true = pd.read_csv('summary_KingsCollege/pose_true.csv')
pose_estim = pd.read_csv('summary_KingsCollege/pose_estim.csv')

position_true = pose_true.iloc[:, 0:7].values
position_estim = pose_estim.iloc[:, 0:7].values

fig = plt.figure()
ax = fig.gca(projection='3d')
# plt.set_xlabel('x')
# plt.set_ylabel('y')
# ax.set_zlabel('z')

ax.scatter(position_true[:, 0], position_true[:, 1], position_true[:, 2], c='r', marker='o', label='Truth')
ax.scatter(position_estim[:, 0], position_estim[:, 1], position_estim[:, 2], c='b', marker='o', label='estimation')
#
# plt.scatter(position_true[:, 6], position_true[:, 1], c='r', marker='o', label='Truth')
# plt.scatter(position_estim[:, 6], position_estim[:, 1], c='b', marker='o', label='estimation')

plt.legend()

# plt.axis('scaled')
plt.show()
