import cv2
import joblib
from skimage.feature import hog
import numpy
from splice import splice_image
from PIL import Image



def predict_single(image):
    clf = joblib.load("digits_cls.pkl")
    im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
    ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)
    ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]
    # nbr = 1

    for rect in rects:
        #cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
        leng = int(rect[3] * 1.6)
        pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
        pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
        roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
        height, width = roi.shape
        if height != 0 and width != 0:
            roi = cv2.resize(roi, (80, 80), interpolation=cv2.INTER_AREA)
            roi = cv2.dilate(roi, (3, 3))
            roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(40, 40), cells_per_block=(1, 1), visualize=False)
            nbr = clf.predict(numpy.array([roi_hog_fd], 'float64'))
            return (nbr[0])
            
def predict_grid(image):
    imgArr = splice_image(image)
    grid = []
    for i in range(81):
        grid.append(predict_single(imgArr[i]))
    #print(len(grid))
    #print(grid)

    grid2 = [[0 for i in range(9)] for j in range(9)]
    counter = 0
    for i in range(81):
        if grid[i] is None:
            grid[i] = int(0)
        else:
            grid[i] = int(grid[i])
    counter = 0
    print(grid)
    for i in range(9):
        for j in range(9):
            grid2[i][j] = grid[counter]
            counter = counter + 1
            

    


    return grid2

if __name__ == '__main__':
    img = cv2.imread('images/download3.png')
    # img = cv2.resize(img, dsize=(40, 40), interpolation=cv2.INTER_CUBIC)
    print(predict_grid(img))

