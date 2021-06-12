import numpy as np  
   
# 计算脸部特征比例
def face_features(points):
    face_width = []
    face_length = []
    face_rate = []
    yi9 = points[8,1]
    for i in range(8):
        fw = points[i, 0] - points[16-i, 0]
        face_width.append(fw)
        yi = (points[i,1] + points[16-i,1])/2
        fl = yi - yi9
        face_length.append(fl)
    for j in range(8):
        fr = face_length[j] / face_width[j]
        face_rate.append(fr)
    return face_rate

# 计算眼部特征比例
def eye_features(points):
    eye_length_right = points[39, 0] - points[36, 0]
    eye_length_left = points[47, 0] - points[42, 0]
    eye_width_right = max((points[47, 1] - points[43, 1]), (points[46, 1] - points[44, 1])) 
    eye_width_left = max((points[41, 1] - points[37, 1]), (points[40, 1] - points[48, 1]) )
    eye_rate_right = eye_length_right / eye_width_right
    eye_rate_left = eye_length_left / eye_width_left
    face_width = points[16, 0] - points[0, 0]
    face_eye_rate = max(eye_length_left, eye_length_right) / face_width
    eye2eye = points[42, 0] -points[39, 0]
    eyewithface = eye2eye / face_width

    return eye_rate_left, eye_rate_right, eyewithface, face_eye_rate

# 计算嘴部特征比例
def mouth_feature(points):
    mouth_length = points[54, 0] - points[48, 0] 
    mouth_width = points[57, 1] - min(points[50, 1], points[52, 1])
    mouth_rate = mouth_length / mouth_width
    lowlip_width = points[57, 1] - points[66, 1]
    face_width = points[11, 0] - points[5, 0]
    face_length = points[8, 1] - points[0, 1]
    mouth_face_rate1 = mouth_length / face_width
    mouth_face_rate2 = mouth_width / face_length
    lip_width_rate = lowlip_width / mouth_width
    
    return mouth_rate, mouth_face_rate1, mouth_face_rate2, lip_width_rate

def face_rate2(points):
    # 长为眉角到嘴角的距离
    brow_pos = (points[17, 1] + points[26, 1]) / 2
    lip_pos = (points[48, 1] + points[54, 1]) / 2
    length = lip_pos - brow_pos
    # 宽为两边腮骨的距离
    width = points[11, 0] - points[5, 0]
    # 长宽比
    face_rate = length / width
    return face_rate

