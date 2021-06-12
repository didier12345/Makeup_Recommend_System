import dlib         # 人脸识别的库 Dlib
import numpy as np  # 数据处理的库 numpy
import cv2          # 图像处理的库 OpenCv
from detector import face_detector
from image_cut import cutpicture
from predictor import face_predictor,face_makeup_recommender

# Dlib 检测器和预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# 读取图像文件
img_rd = cv2.imread("test1.png")



cv2.imshow("image", img_rd)


all_pos,local = face_detector(img_rd,detector,predictor)

# 脸型预测
predict_face = face_predictor(local[0])
print(predict_face)
# 相应推荐妆容
face_makeup = face_makeup_recommender(predict_face)
print(face_makeup)

face_pos = cutpicture(img_rd,all_pos[0])
eyebrow_pos1 = cutpicture(img_rd,all_pos[1])
eyebrow_pos2 = cutpicture(img_rd,all_pos[2])
nose_pos = cutpicture(img_rd,all_pos[3])
eye_pos1 = cutpicture(img_rd,all_pos[4])
eye_pos2 = cutpicture(img_rd,all_pos[5])
mouth_pos = cutpicture(img_rd,all_pos[6])
lip_pos = cutpicture(img_rd,all_pos[7])

cv2.imshow("image1", face_pos)
cv2.imshow("image2", eyebrow_pos1)
cv2.imshow("image3", eyebrow_pos2)
cv2.imshow("image4", nose_pos)
cv2.imshow("image6", eye_pos1)
cv2.imshow("image7", eye_pos2)
cv2.imshow("image8", mouth_pos)
cv2.imshow("image9", lip_pos)

cv2.waitKey(0)