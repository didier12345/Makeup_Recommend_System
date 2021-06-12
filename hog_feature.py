from skimage.feature import hog
from skimage import io
from skimage.color import rgb2gray
from PIL import Image
import cv2
import numpy
 

img1 = cv2.imread('test1.png')
#print (img.shape)
#gray = rgb2gray(img1)
#print (gray.shape)
fd1, hog_image1 = hog(img1, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(16, 16), block_norm='L2-Hys',visualize=True)
img2 = io.imread('test2.png')
img3 = io.imread('test3.png')
img4 = io.imread('test4.png')
img6 = io.imread('test5.png')
img5 = io.imread('test6.jpg')

fd2, hog_image2 = hog(img2, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(16, 16), block_norm='L2-Hys',visualize=True)
fd3, hog_image3 = hog(img3, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(16, 16), block_norm='L2-Hys',visualize=True)
fd4, hog_image3 = hog(img4, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(16, 16), block_norm='L2-Hys',visualize=True)
fd5, hog_image3 = hog(img5, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(16, 16), block_norm='L2-Hys',visualize=True)
fd6, hog_image3 = hog(img6, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(16, 16), block_norm='L2-Hys',visualize=True)

fd61 = fd6 - fd1
fd62 = fd6 - fd2
fd63 = fd6 - fd3
fd64 = fd6 - fd4
fd65 = fd6 - fd5

fd61_norm = (fd61 - numpy.min(fd61)) / (numpy.max(fd61) - numpy.min(fd61))
fd62_norm = (fd62 - numpy.min(fd62)) / (numpy.max(fd62) - numpy.min(fd62))
fd63_norm = (fd63 - numpy.min(fd63)) / (numpy.max(fd63) - numpy.min(fd63))
fd64_norm = (fd64 - numpy.min(fd64)) / (numpy.max(fd64) - numpy.min(fd64))
fd65_norm = (fd65 - numpy.min(fd65)) / (numpy.max(fd65) - numpy.min(fd65))


# fd23 = fd2 - fd3

# square12 = numpy.square(fd12)
# sum12 = numpy.sum(square12)

dist1 = numpy.sqrt(numpy.sum(numpy.square(fd61_norm)))
dist2 = numpy.sqrt(numpy.sum(numpy.square(fd62_norm)))
dist3 = numpy.sqrt(numpy.sum(numpy.square(fd63_norm)))
dist4 = numpy.sqrt(numpy.sum(numpy.square(fd64_norm)))
dist5 = numpy.sqrt(numpy.sum(numpy.square(fd65_norm)))

print(dist1)
print(dist2)
print(dist3)
print(dist4)
print(dist5)


io.imshow(hog_image1)
io.show()
