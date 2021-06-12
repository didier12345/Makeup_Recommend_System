import dlib         # 人脸识别的库 Dlib
import numpy as np  # 数据处理的库 numpy
import cv2          # 图像处理的库 OpenCv
from describer import face_features
from describer import eye_features
from describer import mouth_feature
from describer import face_rate2
from predictor import face_predictor
from predictor import face_makeup_recommender
from compare import comparison

# Dlib 检测器和预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# 读取图像文件
img_rd = cv2.imread("test1.png")
img_gray = cv2.cvtColor(img_rd, cv2.COLOR_RGB2GRAY)

# 人脸数
faces = detector(img_gray, 0)

# 待会要写的字体
font = cv2.FONT_HERSHEY_SIMPLEX

# 特征集
local_features = []

# 标 68 个点
if len(faces) != 0:
    # 检测到人脸
    for i in range(len(faces)):
        # 取特征点坐标
        landmarks = np.matrix([[p.x, p.y] for p in predictor(img_rd, faces[i]).parts()])
        for idx, point in enumerate(landmarks):
            # 68 点的坐标
            pos = (point[0, 0], point[0, 1])

            # 利用 cv2.circle 给每个特征点画一个圈，共 68 个
            cv2.circle(img_rd, pos, 2, color=(139, 0, 0))
            # 利用 cv2.putText 写数字 1-68
            cv2.putText(img_rd, str(idx), pos, font, 0.2, (187, 255, 255), 1, cv2.LINE_AA)
        # 提取人脸特征值
        face_rate = face_features(landmarks)
        face_rate_2 = face_rate2(landmarks)
        local_features.append(face_rate)
        for idx, rate in enumerate(face_rate):
            rate_pos = (10, 260+idx*10)
            cv2.putText(img_rd, 'face rate'+str(idx)+' : '+str(round(rate,2)), rate_pos, font, 0.2, (0, 0, 0), 1, cv2.LINE_AA)
        # 提取眼睛特征值
        eye_rate = eye_features(landmarks)
        local_features.append(eye_rate)
        cv2.putText(img_rd, 'left eye rate : '+str(round(eye_rate[0],2)), (250, 300), font, 0.2, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img_rd, 'right eye rate : '+str(round(eye_rate[1],2)), (250, 310), font, 0.2, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img_rd, 'face eye rate : '+str(round(eye_rate[2],2)), (250, 320), font, 0.2, (0, 0, 0), 1, cv2.LINE_AA)
        # 提取嘴巴特征值
        mouth_rate = mouth_feature(landmarks)
        local_features.append(mouth_rate)
        cv2.putText(img_rd, 'mouth rate : '+str(round(mouth_rate[0],2)), (250, 340), font, 0.2, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.putText(img_rd, "faces: " + str(len(faces)), (20, 50), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
else:
    # 没有检测到人脸
    cv2.putText(img_rd, "no face", (20, 50), font, 1, (0, 0, 0), 1, cv2.LINE_AA)

print(face_rate_2)
# 脸型预测
predict_face = face_predictor(face_rate_2)
print(predict_face)
# 相应推荐妆容
face_makeup = face_makeup_recommender(predict_face)
print(face_makeup)

# 取第二张图片作比较
img_rd_2 = cv2.imread("test3.png")
img_gray_2 = cv2.cvtColor(img_rd_2, cv2.COLOR_RGB2GRAY)

faces_2 = detector(img_gray_2, 0)
# 第二张图片特征集
local_features_2 = []

if len(faces_2) != 0:
    for i in range(len(faces_2)):
        landmarks_2 = np.matrix([[p.x, p.y] for p in predictor(img_rd_2, faces_2[i]).parts()])

local_features_2.append(face_features(landmarks_2))
local_features_2.append(eye_features(landmarks_2))
local_features_2.append(mouth_feature(landmarks_2))

similarity = comparison(local_features, local_features_2)
print('face similarity:' + format(similarity[0], '.2f'))
print('eye similarity:' + format(similarity[1], '.2f'))
print('mouth similarity:' + format(similarity[2], '.2f'))

# 窗口显示
# 参数取 0 可以拖动缩放窗口，为 1 不可以
# cv2.namedWindow("image", 0)

cv2.imshow("image", img_rd)
