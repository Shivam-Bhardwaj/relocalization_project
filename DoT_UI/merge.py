test = open("image_name.txt", "r").readlines()
result = open("result.txt", "r").readlines()
new_test = open("test_.txt", "w")

names = []
pose = []
for i in range(len(result)):
    names.append((test[i].split("/")[-1]).split())
    pose.append((result[i].split()))

import numpy as np
a = np.asarray(names)
b_old = np.asarray(pose)

b = np.delete(b_old, 0, axis=1)
c = np.hstack((a, b))
# print(c[1])
s = ''
for i in c:
    for j in i:
        s += j + ' '
    s += '\n'

new_test.writelines(s)
