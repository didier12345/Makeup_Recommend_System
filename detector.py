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
from image_cut import cutpicture


def face_detector(img_rd,detector,predictor,show_track = False,show_point = False):

    whole_pos = []
    # face_pos = []
    # eyebrow_pos1 = []
    # eyebrow_pos2 = []
    # nose_pos1 = []
    # nose_pos2 = []
    # eye_pos1 = []
    # eye_pos2 = []
    # mouth_pos = []
    # lip_pos = []

    img_gray = cv2.cvtColor(img_rd, cv2.COLOR_RGB2GRAY)
    
    img_cpoy = img_rd.copy()
    # 人脸数
    faces = detector(img_gray, 0)
    # 特征集
    local_features = []
    # 待会要写的字体
    font = cv2.FONT_HERSHEY_SIMPLEX
    # 标 68 个点
    if len(faces) != 0:
        # 检测到人脸
        for i in range(len(faces)):
            # 取特征点坐标
            landmarks = np.matrix([[p.x, p.y] for p in predictor(img_rd, faces[i]).parts()])
            if show_point == True:
                for idx, point in enumerate(landmarks):
                    # 68 点的坐标
                    pos = (point[0, 0], point[0, 1])
                    # 利用 cv2.circle 给每个特征点画一个圈，共 68 个
                    cv2.circle(img_rd, pos, 2, color=(139, 0, 0))
                    # 利用 cv2.putText 写数字 1-68
                    cv2.putText(img_rd, str(idx), pos, font, 0.2, (187, 255, 255), 1, cv2.LINE_AA)
            face_pos = []
            eyebrow_pos1 = []
            eyebrow_pos2 = []
            nose_pos = []
            eye_pos1 = []
            eye_pos2 = []
            mouth_pos = []
            lip_pos = []
            for index in range(68):
                # 68 点的坐标
                pos = (landmarks[index,0], landmarks[index,1])
                # 脸轮廓
                if index <= 16:
                    face_pos.append(pos)
                if index > 16 and index <= 21:
                    eyebrow_pos1.append(pos)
                if index > 21 and index <= 26:
                    eyebrow_pos2.append(pos)
                if index > 26 and index <= 35:
                    nose_pos.append(pos)
                if index > 35 and index <= 41:
                    eye_pos1.append(pos)
                if index > 41 and index <= 47:
                    eye_pos2.append(pos)
                if index > 47 and index <= 59:
                    mouth_pos.append(pos)
                if index > 59 and index <= 67:
                    lip_pos.append(pos)
            
            whole_pos.append(face_pos)
            whole_pos.append(eyebrow_pos1)
            whole_pos.append(eyebrow_pos2)
            whole_pos.append(nose_pos)
            whole_pos.append(eye_pos1)
            whole_pos.append(eye_pos2)
            whole_pos.append(mouth_pos)
            whole_pos.append(lip_pos)   
            
            if show_track == True:
                # 画脸
                for index,point in enumerate(face_pos):
                    if index == 0:
                        first_point = point
                        temp_point = point
                    # elif index == len(face_pos) - 1:
                    #     cv2.line(img_rd,point,temp_point,(0,0,0),1)
                    #     cv2.line(img_rd,point,first_point,(0,0,0),1)
                    else:
                        cv2.line(img_rd,point,temp_point,(187, 255, 255),1)
                        temp_point = point
                # 画眉1
                for index,point in enumerate(eyebrow_pos1):
                    if index == 0:
                        first_point = point
                        temp_point = point
                    elif index == len(eyebrow_pos1) - 1:
                        cv2.line(img_rd,point,temp_point,(187, 255, 255),1)
                        cv2.line(img_rd,point,first_point,(187, 255, 255),1)
                    else:
                        cv2.line(img_rd,point,temp_point,(187, 255, 255),1)
                        temp_point = point
                # 画眉2
                for index,point in enumerate(eyebrow_pos2):
                    if index == 0:
                        first_point = point
                        temp_point = point
                    elif index == len(eyebrow_pos2) - 1:
                        cv2.line(img_rd,point,temp_point,(187, 255, 255),1)
                        cv2.line(img_rd,point,first_point,(187, 255, 255),1)
                    else:
                        cv2.line(img_rd,point,temp_point,(187, 255, 255),1)
                        temp_point = point
                # 画鼻
                for index,point in enumerate(nose_pos):
                    if index == 0:
                        first_point = point
                        temp_point = point
                    elif index == len(nose_pos) - 1:
                        cv2.line(img_rd,point,temp_point,(187, 255, 255),1)
                        cv2.line(img_rd,point,first_point,(187, 255, 255),1)
                    else:
                        cv2.line(img_rd,point,temp_point,(187, 255, 255),1)
                        temp_point = point
                # # 画鼻2
                # for index,point in enumerate(nose_pos2):
                #     if index == 0:
                #         first_point = point
                #         temp_point = point
                #     elif index == len(nose_pos2) - 1:
                #         cv2.line(img_rd,point,temp_point,(187, 255, 255),1)
                #         cv2.line(img_rd,point,first_point,(187, 255, 255),1)
                #     else:
                #         cv2.line(img_rd,point,temp_point,(187, 255, 255),1)
                #         temp_point = point
                # 画眼1
                for index,point in enumerate(eye_pos1):
                    if index == 0:
                        first_point = point
                        temp_point = point
                    elif index == len(eye_pos1) - 1:
                        cv2.line(img_rd,point,temp_point,(187, 255, 255),1)
                        cv2.line(img_rd,point,first_point,(187, 255, 255),1)
                    else:
                        cv2.line(img_rd,point,temp_point,(187, 255, 255),1)
                        temp_point = point
                # 画眼2
                for index,point in enumerate(eye_pos2):
                    if index == 0:
                        first_point = point
                        temp_point = point
                    elif index == len(eye_pos2) - 1:
                        cv2.line(img_rd,point,temp_point,(187, 255, 255),1)
                        cv2.line(img_rd,point,first_point,(187, 255, 255),1)
                    else:
                        cv2.line(img_rd,point,temp_point,(187, 255, 255),1)
                        temp_point = point
                # 画嘴
                for index,point in enumerate(mouth_pos):
                    if index == 0:
                        first_point = point
                        temp_point = point
                    elif index == len(mouth_pos) - 1:
                        cv2.line(img_rd,point,temp_point,(187, 255, 255),1)
                        cv2.line(img_rd,point,first_point,(187, 255, 255),1)
                    else:
                        cv2.line(img_rd,point,temp_point,(187, 255, 255),1)
                        temp_point = point
                # 画唇
                for index,point in enumerate(lip_pos):
                    if index == 0:
                        first_point = point
                        temp_point = point
                    elif index == len(lip_pos) - 1:
                        cv2.line(img_rd,point,temp_point,(187, 255, 255),1)
                        cv2.line(img_rd,point,first_point,(187, 255, 255),1)
                    else:
                        cv2.line(img_rd,point,temp_point,(187, 255, 255),1)
                        temp_point = point
            # # 利用 cv2.circle 给每个特征点画一个圈，共 68 个
            # cv2.circle(img_rd, pos, 2, color=(139, 0, 0))
            # # 利用 cv2.putText 写数字 1-68
            # cv2.putText(img_rd, str(idx), pos, font, 0.2, (187, 255, 255), 1, cv2.LINE_AA)
            # temp_pos = pos
        # 提取人脸特征值
            # face_rate = face_features(landmarks)
            face_rate_2 = face_rate2(landmarks)
            local_features.append(face_rate_2)
            # for idx, rate in enumerate(face_rate):
            #     rate_pos = (10, 260+idx*10)
                # cv2.putText(img_rd, 'face rate'+str(idx)+' : '+str(round(rate,2)), rate_pos, font, 0.2, (0, 0, 0), 1, cv2.LINE_AA)
            # 提取眼睛特征值
            eye_rate = eye_features(landmarks)
            local_features.append(eye_rate)
            # cv2.putText(img_rd, 'left eye rate : '+str(round(eye_rate[0],2)), (250, 300), font, 0.2, (0, 0, 0), 1, cv2.LINE_AA)
            # cv2.putText(img_rd, 'right eye rate : '+str(round(eye_rate[1],2)), (250, 310), font, 0.2, (0, 0, 0), 1, cv2.LINE_AA)
            # cv2.putText(img_rd, 'face eye rate : '+str(round(eye_rate[2],2)), (250, 320), font, 0.2, (0, 0, 0), 1, cv2.LINE_AA)
            # 提取嘴巴特征值
            mouth_rate = mouth_feature(landmarks)
            local_features.append(mouth_rate)
            # cv2.putText(img_rd, 'mouth rate : '+str(round(mouth_rate[0],2)), (250, 340), font, 0.2, (0, 0, 0), 1, cv2.LINE_AA)

        cv2.putText(img_rd, "faces: " + str(len(faces)), (20, 50), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
    else:
        # 没有检测到人脸
        cv2.putText(img_rd, "no face", (20, 50), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
    return whole_pos, local_features
