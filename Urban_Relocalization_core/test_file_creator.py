import os

image_name = os.listdir("images")
# print(image_name)

with open('test.txt', 'w') as filehandle:
    for listitem in image_name:
        filehandle.write('%s 0 0 0 0 0 0 0\n' % listitem)