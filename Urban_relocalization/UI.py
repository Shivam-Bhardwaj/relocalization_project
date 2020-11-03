import cv2
import numpy as np    
import os
import time

class UI:
    def __init__(self):
        self.width = 640
        self.height = 480
        self.img1 = self.getBlankImage(self.width,self.height)
        self.img2 = self.getBlankImage(self.width,self.height)
        self.img3 = self.getBlankImage(self.width,self.height*2+20)
        self.cap = cv2.VideoCapture(0)

    def getBlankImage(self,width,height):
        blank_image = np.ones((height,width,3), np.uint8)*255
        blank_image = self.borderImage(blank_image)
        return blank_image

    def borderImage(self,img):
        top = 10  
        bottom = top
        left = 10  
        right = left
        result = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, None, 0)
        return result

    def begin(self):
        image_list = os.listdir("./data/dataset_house/train/")
        image_list = sorted(image_list)
        for image in image_list[:-1]:
            time.sleep(0.05)
            self.img1 = cv2.imread("./data/dataset_house/train/" + image)
                # img1=cv2.imread("C:/Users/Smit/Desktop/AI4CE/NYC_DOT_Navigator/Shivam/assets/images/000510.jpg")
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

            self.img1 = cv2.resize(self.img1,(self.width,self.height))
            self.img1 = self.borderImage(self.img1)
                
            stack1 = np.vstack((self.img1,self.img2))
            disp = np.hstack((stack1,self.img3))
            cv2.imshow("Output Window",disp)
            if cv2.waitKey(1)& 0xFF == ord('q'):
                break
        self.cap.release()    
        cv2.destroyAllWindows()

ui = UI()
ui.begin()