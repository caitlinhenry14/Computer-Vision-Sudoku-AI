import io
from splice import splice_image
import numpy
from flask import Flask, render_template, request, redirect, url_for, flash
import solve
from solve import return_grid
import solvable
from solvable import isValid
import subprocess
import os
import cv2
from werkzeug.utils import secure_filename
from predict import predict_grid
import numpy as np


app = Flask(__name__, template_folder='templates')
import tensorflow as tf
import keras
from keras.models import load_model
tf.logging.set_verbosity(tf.logging.ERROR)


model = load_model('digits.h5')

app.config["IMAGE_UPLOADS"] = "/Users/allen/Desktop/SudokuSolver/static/img"
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG", "GIF"]
app.config["MAX_IMAGE_FILESIZE"] = 0.5 * 1024 * 1024

def prepare(img):
	img_size = 28
	new_arr = cv2.resize(img, (img_size, img_size)) 
	new_arr = numpy.invert(new_arr) 

	return new_arr.reshape(-1, img_size, img_size, 1)

def blankSpot(blank):

	s = set()
	for i in range(10):
		for j in range(10):
			#for k in range(2):
			s.add(blank[i + 10][j + 10])
			#print((i + 10), ",", (j+10), ": ", blank[i + 10][j+10])
	#print(s)
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

def CNN_predict_grid(filepath):
	#image_grid = cv2.imread('/Users/allen/Desktop/download.png',cv2.IMREAD_GRAYSCALE)
	image_grid = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
	imgArr = splice_image(image_grid)
	grid = []
	for i in range(81):
		grid.append(CNN_predict_single(imgArr[i]))
	return grid

def allowed_image(filename):

	if not "." in filename:
		return False

	ext = filename.rsplit(".", 1)[1]

	if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
		return True
	else:
		return False


def allowed_image_filesize(filesize):

	if int(filesize) <= app.config["MAX_IMAGE_FILESIZE"]:
		return True
	else:
		return False

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
	return render_template("index.html")

@app.route("/upload", methods=['POST'])
def upload():
	target = os.path.join(APP_ROOT, 'images/')
	print(target)

	if not os.path.isdir(target):
		os.mkdir(target)

	file = request.files.getlist("file")[0]
	# for file in request.files.getlist("file"):
	print("**************************")
	print(type(file))
	print(file)
	filename = file.filename
	print(filename)
	in_memory_file = io.BytesIO()
	file.save(in_memory_file)
	data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
	color_image_flag = 1
	img = cv2.imdecode(data, color_image_flag)
	
	completeArray = predict_grid(img)
   
	return render_template("success.html", nums = completeArray)


@app.route('/return', methods = ['GET', 'POST'])
def returnToHome():
	return render_template("index.html")


@app.route('/send', methods = ['GET' ,'POST'])
def obtainInput():
	grid = [[0 for i in range(9)] for j in range(9)]
	grid2 = [[0 for i in range(9)] for j in range(9)]
	if request.method == 'POST':
		for i in range(9):
			for j in range(9):
				s = ""+str(i) + ","+str(j)
				temp = str(request.form[s])
				if len(temp) == 0:
					temp = str(0);
				grid[j][i] = int(temp)
				grid2[i][j] = int(temp)
		print("DEBUGGING---------------------------------------------")

		print(grid2)

		print(isValid(grid2))
		if isValid(grid2) == False:
			return render_template('fail.html')

		grid = return_grid(grid)
		#print(grid)
		#print(grid2)
		#print("***********")
		hint = []
		for i in range(9):
			for j in range(9):
				if grid[i][j] != grid2[i][j] :
					hint.append(grid[i][j])
		
		#print(hint)
		#print(len(hint))

		print("******************************************************")




	iterate = {0, 1, 2, 3, 4, 5, 6, 7, 8}

	return render_template('pass.html', nums=grid, hints = hint, hLen = len(hint)) #, num = n)


if __name__ == '__main__':
	app.run(debug=True)
