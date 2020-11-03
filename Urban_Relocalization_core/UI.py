import itertools
import os
import time
import cv2
import numpy as np
from sklearn.decomposition import PCA


class UI:
    def __init__(self):
        self.width = 640
        self.height = 480
        self.borderWidth = 10
        self.img1 = self.getBlankImage(self.width, self.height, self.borderWidth)
        self.img2 = self.getBlankImage(self.width, self.height, self.borderWidth)
        self.img3 = self.getBlankImage(self.width, self.height * 2 + self.borderWidth * 2, self.borderWidth)

        self.img3 = self.getBlankImage(self.width, self.height * 2 + self.borderWidth * 2, self.borderWidth)
        print(self.img3.shape)
        shape1_ = self.img3.shape
        self.img3 = cv2.imread("./1.png")
        print(self.img3.shape)
        self.img3 = cv2.resize(self.img3, None, fy=shape1_[0] / self.img3.shape[0], fx=shape1_[1] / self.img3.shape[1],
                               interpolation=cv2.INTER_CUBIC)

    def getBlankImage(self, width, height, borderWidth):
        blank_image = np.ones((height, width, 3), np.uint8) * 255
        blank_image = self.borderImage(blank_image, borderWidth)
        return blank_image

    def borderImage(self, img, borderWidth=10):
        top = borderWidth
        bottom = top
        left = borderWidth
        right = left
        result = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, None, 0)
        return result

    def getScales(self, X):
        self.xmax_value = int(np.ceil((abs(max(X)))))
        self.xmin_value = int(np.ceil((abs(min(X)))))
        self.x_scale = self.xmax_value + self.xmin_value
        self.y_scale = 32

    def drawQuivers(self, X, Y, U, V, length, image, width, height, color):
        for iter in range(len(X)):
            pt1 = (Y[iter] * -1 * (self.y_scale) + width / 2,
                   X[iter] * (height / self.x_scale) * -1 + (height / self.x_scale) * (self.xmax_value))
            pt1 = tuple(map(int, pt1))

            angle = np.arctan2(V[iter], U[iter]) + (1.5708)  # add 1.5708 to rotate by 90 degree
            x2 = int(pt1[0] - length * np.cos(angle))
            y2 = int(pt1[1] + length * np.sin(angle))
            pt2 = (x2, y2)

            try:
                x1 = pt1[0] + 10
                y1 = pt1[1] - 400
                x2 = pt2[0] + 10
                y2 = pt2[1] - 400

                if color[2] == 255:
                    cv2.arrowedLine(image, (x1, y1), (x2, y2 + 2 * (y1 - y2)), color, 3, tipLength=0.3)
                else:
                    cv2.arrowedLine(image, (x1, y1), (x2, y2 + 2 * (y1 - y2)), color, 1, tipLength=0.1)
            except:
                pass

    def qvec2rotmat(self, qvec):
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

    def xyz_rot(self, pos, ori):
        rot = self.qvec2rotmat(ori)
        rot_t = np.transpose(rot)
        rot_neg = -1 * rot_t

        xyz_ = np.dot(rot_neg, pos)
        angle_ = np.dot(rot, [1.0, 1.0, 1.0])
        return xyz_, angle_

    def drawRelativePoseTrain(self, ground_truth, image):
        file_object = open(ground_truth, "r")
        list_from_text_file = file_object.read().split()

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
            xyz_[i], angle_[i] = self.xyz_rot(pos[i], quat[i])

        pca2 = PCA(n_components=2)
        Reduced_pose = np.zeros((n, 2))
        Reduced_rot = np.zeros((n, 2))
        Reduced_pose[:, 0:2] = pca2.fit_transform(xyz_)
        Reduced_rot[:, 0:2] = pca2.fit_transform(angle_)

        frequency = 4
        self.getScales(Reduced_pose[::frequency, 0])
        self.drawQuivers(Reduced_pose[::frequency, 0], Reduced_pose[::frequency, 1], Reduced_rot[::frequency, 0],
                         Reduced_rot[::frequency, 1], 20, image, image.shape[1], image.shape[0], (255, 0, 0))

    def getRelativePoseTest(self, test_file_list):
        pos = []
        quat = []

        for row in test_file_list:
            item1 = row[1:4]
            item1 = list(map(float, item1))
            pos.append(item1)
            item2 = row[4:8]
            item2 = list(map(float, item2))
            quat.append(item2)

        n = len(test_file_list)
        xyz_ = np.zeros((n, 3))
        angle_ = np.zeros((n, 3))

        quat = np.asarray(quat).reshape(n, 4)
        pos = np.asarray(pos).reshape(n, 3)

        for i in range(0, n):
            xyz_[i], angle_[i] = self.xyz_rot(pos[i], quat[i])

        pca2 = PCA(n_components=2)
        Reduced_pose = np.zeros((n, 2))
        Reduced_rot = np.zeros((n, 2))
        Reduced_pose[:, 0:2] = pca2.fit_transform(xyz_)
        Reduced_rot[:, 0:2] = pca2.fit_transform(angle_)

        return Reduced_pose, Reduced_rot

    def readTestFile(self, path):
        file_object = open(path, "r")
        list_from_text_file = file_object.read().split()

        chunks = [list_from_text_file[x:x + 8] for x in range(0, len(list_from_text_file), 8)]
        return chunks

    def readTrainFile(self, path):
        file_object = open(path, "r")
        list_from_text_file = file_object.read().split()

        chunks = [list_from_text_file[x:x + 8] for x in range(0, len(list_from_text_file), 8)]
        return chunks

    def finding_closest_image(self, train_rows, test_row):
        distance_list = list()
        b = test_row[1:8]
        b = list(map(float, b))
        b = np.asarray(b)
        for row in train_rows:
            a = row[1:8]
            a = list(map(float, a))
            a = np.asarray(a)
            dis = np.linalg.norm(a - b)
            distance_list.append(dis)

        sorted_distace_list = sorted(distance_list)
        index = distance_list.index(sorted_distace_list[0])

        return train_rows[index][0]

    def begin(self):
        ground_truth = "./data/dataset_jayst/train/train.txt"
        test_file_path = "./results/test_org/test.txt"
        # test_file_path = "./results/test_result/result.txt"
        train_file_list = self.readTrainFile(ground_truth)
        test_file_list = self.readTestFile(test_file_path)

        image_list = os.listdir("./data/dataset_jayst/test/")
        # image_list = os.listdir("./results/test_result/result.txt")
        image_list = sorted(image_list)

        self.drawRelativePoseTrain(ground_truth, self.img3)
        test_reduced_pose, test_reduced_rot = self.getRelativePoseTest(test_file_list)

        i = 0
        for image in image_list[:-1]:
            time.sleep(0.3)
            # print(i)
            # i+=1
            # cv2.waitKey()

            matched_row = [row for row in test_file_list if image == row[0]]
            matched_row = list(itertools.chain.from_iterable(matched_row))

            index = test_file_list.index(matched_row)

            nearest_image_filename = self.finding_closest_image(train_file_list, matched_row)

            right_img = self.img3.copy()

            self.drawQuivers([test_reduced_pose[index, 0]], [test_reduced_pose[index, 1]], [test_reduced_rot[index, 0]],
                             [test_reduced_rot[index, 1]], 20, right_img, right_img.shape[1], right_img.shape[0],
                             (0, 0, 255))

            self.img1 = cv2.imread("./data/dataset_jayst/test/" + image)
            self.img1 = cv2.resize(self.img1, (self.width, self.height))
            self.img1 = self.borderImage(self.img1)

            self.img2 = cv2.imread("./data/dataset_jayst/train/" + nearest_image_filename)
            self.img2 = cv2.resize(self.img2, (self.width, self.height))
            self.img2 = self.borderImage(self.img2)

            '''
            if cv2.waitKey(1) & 0xFF == ord('s'):  # if key 's' is pressed 
                query_vlad = getting_VLAD_for_Query_image(k_means_codebook_object,img2)
                nearest_images = finding_closest_image_from_database(vlad_descriptors, query_vlad)
                img2=cv2.imread(nearest_images[0])
                img2=cv2.resize(img2,(width,height))
                img2=borderImage(img2)
                angle = -45
                length = 50
                x1 = int(img2.shape[1]/2)
                y1 = int(img2.shape[0]/2)
                x2 =  int(x1 + length * np.cos(angle * 3.14 / 180.0))
                y2 =  int(y1 + length * np.sin(angle * 3.14 / 180.0))
                cv2.arrowedLine(img2, (x1,y1), (x2,y2), (0,0,255), 2,tipLength=0.2 )
            '''
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(self.img1, 'Query Image', (10, 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(self.img2, 'Returned Image', (10, 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
            stack1 = np.vstack((self.img1, self.img2))
            disp = np.hstack((stack1, right_img))
            cv2.imshow("Output Window", disp)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()


ui = UI()
ui.begin()
