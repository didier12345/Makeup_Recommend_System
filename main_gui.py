import dlib 
import numpy
import cv2 
import sys
import os
from skimage.feature import hog

from PyQt5.QtGui import QPainter,QPixmap,QImage
from PyQt5 import QtCore
from PyQt5.QtCore import QTimer
from PyQt5 import QtWidgets
from PyQt5.QtGui import QPixmap,QIcon
from PyQt5.QtWidgets import QFileDialog,QWidget,QMessageBox

from Ui_main_gui import Ui_Form 
from Ui_hog_feature import Ui_hog
from Ui_feature_predictor import Ui_predictor
from detector import face_detector
from predictor import face_makeup_recommender,face_predictor

class mywindow(QWidget, Ui_Form):

    def __init__(self):
        # 初始化
        super(mywindow, self).__init__()
        self.cwd=os.getcwd()
        # 分别初始化类里的setupUi()和retranslateUi()
        self.setupUi(self)
        self.retranslateUi(self)



        self.opencamera = False
        self.show_track = False
        self.show_point = False
        self.pushButton.clicked.connect(self.camerashow)
        self.pushButton_3.clicked.connect(self.closecamera)
        self.pushButton_2.clicked.connect(self.predictorwindow_show)
        self.pushButton_4.clicked.connect(self.hogwindow_show)

        self.hogfeatureDialog = hogfeaturewindow()
        self.predictorDialog = predictorwindow()

        self.points_pos = []
        self.picture = []
        
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    def camerashow(self):
            self.opencamera = True
            video_capture = cv2.VideoCapture(1)
            if video_capture is None or not video_capture.isOpened():
                video_capture = cv2.VideoCapture(0)
            while True:
                if self.opencamera == False:
                    break
                if self.checkBox.isChecked():
                    self.show_point = True
                else:
                    self.show_point = False
                if self.checkBox_2.isChecked():
                    self.show_track = True
                else:
                    self.show_track = False
                ret,img = video_capture.read()
                im = img.copy()
                if ret:
                    self.points_pos, local_features = face_detector(img,self.detector,self.predictor,self.show_track,self.show_point)
                    if not local_features:
                        local_features = [0, (0, 0, 0, 0), (0, 0, 0, 0)]
                    self.label_10.setText(str(round(local_features[0],3)))
                    self.label_11.setText(str(round(local_features[1][0],3)))
                    self.label_12.setText(str(round(local_features[1][1],3)))
                    self.label_13.setText(str(round(local_features[1][2],3)))
                    self.label_14.setText(str(round(local_features[1][3],3)))
                    self.label_15.setText(str(round(local_features[2][0],3)))
                    self.label_16.setText(str(round(local_features[2][1],3)))
                    self.label_17.setText(str(round(local_features[2][2],3)))
                    self.label_19.setText(str(round(local_features[2][3],3)))
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    a = QImage(img, img.shape[1], img.shape[0], QImage.Format_RGB888)
                    self.picture = im
                    self.label.setPixmap(QPixmap.fromImage(a))
                    cv2.waitKey(0)
        # cap = cv2.VideoCapture(0)
        # if not cap.isOpened():
        #     print("Cannot open camera")
        # while True:
        #     # 逐帧捕获
        #     ret, img = cap.read()
        #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #     print('ok')
        #     if not ret:
        #         print("Can't receive frame (stream end?). Exiting ...")
        #         break
        #     else:
        #         # face_detector(img,detector,predictor)
                
        #         a = QImage(img, img.shape[1], img.shape[0], QImage.Format_RGB888)
        #         self.label.setPixmap(QPixmap.fromImage(a))
        #         print(1)

    def closecamera(self):
        self.opencamera = False

    def hogwindow_show(self):
        if self.opencamera == True:
            pic = self.picture
            pic = cv2.resize(pic,(300,300))
            self.hogfeatureDialog.img = pic
            img = cv2.cvtColor(pic, cv2.COLOR_RGB2BGR)
            a = QImage(img, img.shape[1], img.shape[0], QImage.Format_RGB888)
            self.hogfeatureDialog.label.setPixmap(QPixmap.fromImage(a))
        self.hogfeatureDialog.show()
    
    def predictorwindow_show(self):
        if self.opencamera == True:
            pic = self.picture
            self.predictorDialog.img = pic
            img = cv2.cvtColor(pic, cv2.COLOR_RGB2BGR)
            a = QImage(img, img.shape[1], img.shape[0], QImage.Format_RGB888)
            self.predictorDialog.label.setPixmap(QPixmap.fromImage(a))
        self.predictorDialog.show()


class hogfeaturewindow(QWidget, Ui_hog):
    def __init__(self):
        super(hogfeaturewindow, self).__init__()
        self.setupUi(self)
        self.retranslateUi(self)

        self.pushButton.clicked.connect(self.calculate)

        self.comboBox.clear()  # 清空items
        self.comboBox.addItems(['请选择图片','图片1','图片2','图片3','图片4','图片5','图片6','图片7'])
        self.comboBox.currentIndexChanged.connect(self.selectionchange1)

        self.comboBox_2.clear()  # 清空items
        self.comboBox_2.addItems(['请选择图片','图片1','图片2','图片3','图片4','图片5','图片6','图片7'])
        self.comboBox_2.currentIndexChanged.connect(self.selectionchange2)

        self.comboBox_3.clear()  # 清空items
        self.comboBox_3.addItems(['请选择图片','图片1','图片2','图片3','图片4','图片5','图片6','图片7'])
        self.comboBox_3.currentIndexChanged.connect(self.selectionchange3)

        self.comboBox_4.clear()  # 清空items
        self.comboBox_4.addItems(['请选择图片','图片1','图片2','图片3','图片4','图片5','图片6','图片7'])
        self.comboBox_4.currentIndexChanged.connect(self.selectionchange4)

        self.background_path = 'test1.png'
        img = cv2.imread(self.background_path)
        a = self.transform(img)
        self.img = img
        self.label.setPixmap(QPixmap.fromImage(a))      

    def transform(self,img):
        img = cv2.resize(img,(300,300))
        img_trans = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        a = QImage(img_trans, img_trans.shape[1], img_trans.shape[0], QImage.Format_RGB888)
        return a

    def calculate(self):
        fd1, hog_image1 = hog(self.img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(16, 16), block_norm='L2-Hys',visualize=True)
        fd2, hog_image2 = hog(self.img1, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(16, 16), block_norm='L2-Hys',visualize=True)
        fd3, hog_image3 = hog(self.img2, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(16, 16), block_norm='L2-Hys',visualize=True)
        fd4, hog_image3 = hog(self.img3, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(16, 16), block_norm='L2-Hys',visualize=True)
        fd5, hog_image3 = hog(self.img4, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(16, 16), block_norm='L2-Hys',visualize=True)
        fd61 = fd1 - fd2
        fd62 = fd1 - fd3
        fd63 = fd1 - fd4
        fd64 = fd1 - fd5
        fd61_norm = (fd61 - numpy.min(fd61)) / (numpy.max(fd61) - numpy.min(fd61))
        fd62_norm = (fd62 - numpy.min(fd62)) / (numpy.max(fd62) - numpy.min(fd62))
        fd63_norm = (fd63 - numpy.min(fd63)) / (numpy.max(fd63) - numpy.min(fd63))
        fd64_norm = (fd64 - numpy.min(fd64)) / (numpy.max(fd64) - numpy.min(fd64))
        dist1 = numpy.sqrt(numpy.sum(numpy.square(fd61_norm)))
        dist2 = numpy.sqrt(numpy.sum(numpy.square(fd62_norm)))
        dist3 = numpy.sqrt(numpy.sum(numpy.square(fd63_norm)))
        dist4 = numpy.sqrt(numpy.sum(numpy.square(fd64_norm)))
        self.label_7.setText(str(round(dist1,2)))
        self.label_8.setText(str(round(dist2,2)))
        self.label_9.setText(str(round(dist3,2)))
        self.label_10.setText(str(round(dist4,2)))

    def hogfeature(self,img):
        fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(16, 16), block_norm='L2-Hys',visualize=True)
        return fd, hog_image

    def selectionchange1(self):
        print(self.comboBox.currentIndex())
        background_index = self.comboBox.currentIndex()
        background_path = [ '',
                            'test1.png',
                            'test2.png',
                            'test3.png',
                            'test4.png',
                            'test5.png',
                            'test6.jpg',
                            'test7.jpg']
        self.background_path = background_path[background_index]
        img = cv2.imread(self.background_path)
        self.img1 = img
        a = self.transform(img)
        self.label_3.setPixmap(QPixmap.fromImage(a))

    def selectionchange2(self):
        print(self.comboBox_2.currentIndex())
        background_index = self.comboBox_2.currentIndex()
        background_path = [ '',
                            'test1.png',
                            'test2.png',
                            'test3.png',
                            'test4.png',
                            'test5.png',
                            'test6.jpg',
                            'test7.jpg']
        self.background_path = background_path[background_index]
        img = cv2.imread(self.background_path)
        self.img2 = img
        a = self.transform(img)
        self.label_4.setPixmap(QPixmap.fromImage(a))

    def selectionchange3(self):
        print(self.comboBox_3.currentIndex())
        background_index = self.comboBox_3.currentIndex()
        background_path = [ '',
                            'test1.png',
                            'test2.png',
                            'test3.png',
                            'test4.png',
                            'test5.png',
                            'test6.jpg',
                            'test7.jpg']
        self.background_path = background_path[background_index]
        img = cv2.imread(self.background_path)
        self.img3 = img
        a = self.transform(img)
        self.label_5.setPixmap(QPixmap.fromImage(a))

    def selectionchange4(self):
        print(self.comboBox_4.currentIndex())
        background_index = self.comboBox_4.currentIndex()
        background_path = [ '',
                            'test1.png',
                            'test2.png',
                            'test3.png',
                            'test4.png',
                            'test5.png',
                            'test6.jpg',
                            'test7.jpg']
        self.background_path = background_path[background_index]
        img = cv2.imread(self.background_path)
        self.img4 = img
        a = self.transform(img)
        self.label_6.setPixmap(QPixmap.fromImage(a))


class predictorwindow(QWidget, Ui_predictor):
    def __init__(self):
        super(predictorwindow, self).__init__()
        self.setupUi(self)
        self.retranslateUi(self)

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

        self.pushButton.clicked.connect(self.showcut)
        self.pushButton_2.clicked.connect(self.predict_recommend)

        # self.pos = []
        # self.local_features = []
        self.background_path = 'test1.png'
        img = cv2.imread(self.background_path)
        self.img = img
        a = self.transform(img)
        self.label_4.setPixmap(QPixmap.fromImage(a)) 
        self.pos, self.local_features = face_detector(img,self.detector,self.predictor)

    def predict_recommend(self):
        # 脸型预测
        predict_face = face_predictor(self.local_features[0])
        self.label_2.setText(predict_face)
        # 相应推荐妆容
        face_makeup = face_makeup_recommender(predict_face)
        self.label_12.setText(face_makeup)

    def transform(self,img):
        img = cv2.resize(img,(300,300))
        img_trans = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        a = QImage(img_trans, img_trans.shape[1], img_trans.shape[0], QImage.Format_RGB888)
        return a

    def showcut(self):
        all_pos, local_features = self.pos,self.local_features
        img = self.img
        face_pos = self.cutpicture(img,all_pos[0])
        eyebrow_pos1 = self.cutpicture(img,all_pos[1])
        eyebrow_pos2 = self.cutpicture(img,all_pos[2])
        nose_pos = self.cutpicture(img,all_pos[3])
        eye_pos1 = self.cutpicture(img,all_pos[4])
        eye_pos2 = self.cutpicture(img,all_pos[5])
        mouth_pos = self.cutpicture(img,all_pos[6])
        a1 = self.transform(face_pos)
        a2 = self.transform(eyebrow_pos1)
        a3 = self.transform(eyebrow_pos2)
        a4 = self.transform(nose_pos)
        a5 = self.transform(eye_pos1)
        a6 = self.transform(eye_pos2)
        a7 = self.transform(mouth_pos)
        self.label.setPixmap(QPixmap.fromImage(a1)) 
        self.label_5.setPixmap(QPixmap.fromImage(a2)) 
        self.label_6.setPixmap(QPixmap.fromImage(a3)) 
        self.label_7.setPixmap(QPixmap.fromImage(a4)) 
        self.label_8.setPixmap(QPixmap.fromImage(a5)) 
        self.label_9.setPixmap(QPixmap.fromImage(a6)) 
        self.label_10.setPixmap(QPixmap.fromImage(a7)) 

    def cutpicture(image,points):
        x_pos = [0,100]
        y_pos = [0,100]
        if points :
            x_pos = []
            y_pos = []
            for point in points:
                x_pos.append(point[0])
                y_pos.append(point[1])
        min_x_pos = abs(min(x_pos) - 10)
        max_x_pos = abs(max(x_pos) + 10)
        min_y_pos = abs(min(y_pos) - 10)
        max_y_pos = abs(max(y_pos) + 10)
        image1 = image[min_y_pos:max_y_pos, min_x_pos:max_x_pos]
        image1 = cv2.resize(image,(300,300))
        return image1

if __name__=="__main__":
    app=QtWidgets.QApplication(sys.argv)
    mywin=mywindow()
    # camerawin=camerawindow()
    # checkwin=checkwindow()
    # ifcamerawin=ifcamerawindow()
    mywin.show()
    sys.exit(app.exec_())