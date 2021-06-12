import cv2

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
    image = image[min_y_pos:max_y_pos, min_x_pos:max_x_pos]
    # image = cv2.resize(image,(300,300))
    return image
    
    # cv2.imshow('face', image)
    # cv2.waitKey(0)
    
    # print(min_x_pos,max_x_pos)
    
