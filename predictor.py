import numpy as np  
  
# 脸型预测
def face_predictor(face_rate):
    face_shape = ['圆脸', '鹅蛋脸', '长脸']
    temp_rate = [face_rate-2/3, face_rate-1, face_rate-3/2]
    rate = list(map(abs, temp_rate))
    index = rate.index(min(rate))
    predict_face = face_shape[index]
    return predict_face

# 脸型对应妆容推荐
def face_makeup_recommender(face_shape):
    makeup_recommender = {'round face':'两腮与额头两边加深色粉底，以长线条方式在脸中部加亮色粉底。\n眉妆使用上扬眉',\
                          'egg face':'好极了，不用化妆\n眉妆使用平眉',\
                          'long face':'额头上方、下巴尖涂深色粉底；额头两侧，下颚角涂亮色粉底。'}
    recomender = makeup_recommender.get(face_shape, 'No suitable makeup.')
    return recomender

# 眼型预测
def eye_predictor(eye_rate):
    eye_shape = ['杏眼', '圆眼', '细长眼']

# 眼型对应妆容推荐
def eye_makeup_recommender(eye_shape):
    makeup_recommender = {}

# 嘴型预测
def mouth_predictor(mouth_rate):
    mouth_shape = []

# 嘴型对应妆容推荐
def mouth_makeup_recommender(mouth_shape):
    makeup_recommender = {}