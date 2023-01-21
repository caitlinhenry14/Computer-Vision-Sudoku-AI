from keras.models import load_model
from keras.datasets import mnist
import matplotlib

import matplotlib.pyplot as plt
from splice import splice_image
import numpy
tf.logging.set_verbosity(tf.logging.ERROR)

model = load_model('digits.h5')
import cv2
#odel.summary()


def prepare(img):
	img_size = 28
	#img_arr = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
	new_arr = cv2.resize(img, (img_size, img_size)) #resize for CNN
	new_arr = numpy.invert(new_arr) #mnist dataset has data that is inverted so we must invert our input
	#plt.imshow(new_arr)
	#plt.show()
	return new_arr.reshape(-1, img_size, img_size, 1)


'''
(X_train, y_train), (X_test, y_test) = mnist.load_data()
#plt.imshow(X_train[1], cmap="gray")
#plt.show()
img = X_train[1]
img = img.reshape(-1, 28, 28, 1)
'''

def blankSpot(blank):

	s = set()
	for i in range(10):
		for j in range(10):
			#for k in range(2):
			s.add(blank[i + 10][j + 10])
	if len(s) == 1:
		return True
	else: 
		return False

def CNN_predict_single(image):
	if blankSpot(image) == True:
		return int(0)
	else:
		prediction = model.predict_classes(prepare(image))
		return(prediction[0])
'''
def CNN_predict_single(image):
	print(blankSpot(image))
	prediction = model.predict_classes(prepare(image))
	return(prediction[0])
'''
def CNN_predict_grid(filepath):
	image_grid = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
	imgArr = splice_image(image_grid)
	grid = []
	for i in range(81):
		grid.append(CNN_predict_single(imgArr[i]))
	return grid
def hello():
    return 'Hi'
