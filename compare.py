import numpy as np  

def comparison(old_rate, new_rate):
    face_similarity = 0
    eye_similarity = 0
    mouth_similarity = 0
    # 脸相似度
    for i in range(len(old_rate[0])):
        temp_similarity = old_rate[0][i] / new_rate[0][i]
        face_similarity += temp_similarity
    face_similarity = face_similarity / len(old_rate[0])
    if face_similarity > 1:
        face_similarity = 1 / face_similarity

    # 眼睛相似度
    for i in range(len(old_rate[1])):
        temp_similarity = old_rate[1][i] / new_rate[1][i]
        eye_similarity += temp_similarity
    eye_similarity = eye_similarity / len(old_rate[0])
    if eye_similarity > 1:
        eye_similarity = 1 / eye_similarity

    # 嘴相似度
    for i in range(len(old_rate[2])):
        temp_similarity = old_rate[2][i] / new_rate[2][i]
        mouth_similarity += temp_similarity
    mouth_similarity = mouth_similarity / len(old_rate[0])
    if mouth_similarity > 1:
        mouth_similarity = 1 / mouth_similarity

    return face_similarity, eye_similarity, mouth_similarity
