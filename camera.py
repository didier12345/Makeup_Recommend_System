import dlib 
import numpy as np
import cv2 
from detector import face_detector
from image_cut import cutpicture

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

stop = False


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    if cv2.waitKey(1) == ord('y'):
        cap = cv2.VideoCapture(0)
        stop = False
    if cv2.waitKey(1) == ord('n'):
        stop = True
        cap.release()
    if (stop == False ):
    # 逐帧捕获
        ret, img = cap.read()
        # img = cv2.resize(img,(480,640))
    # 如果正确读取帧，ret为True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    else:
        print(img.shape)
        face_detector(img,detector,predictor)
        # cutpicture(img_copy,face)
        # image_cut = img[0:100, 0:100]
        cv2.imshow('image', img)
        # print(face_image)
        
    print(stop)
    if cv2.waitKey(1) == ord('q'):
        break
# 完成所有操作后，释放捕获器
cap.release()
cv2.destroyAllWindows()
