import matplotlib.image as mpimg
from skimage.feature import hog
import cv2
import numpy as np
import matplotlib.pyplot as plt 

def get_HOG(img):
	fd, hog_image = hog(img, orientations=9, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, multichannel=True)
	return hog_image

def get_LAB(img):
	LAB_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
	return LAB_image

def cross_neighbourhood_voting(BoW,x):

	for i in superpixels:
		sum = 0
		for k in neighbourhood(i):
			sum += exp(- (BoW[i] - BoW[k])^2)
		neigh = 0
		for k in neighbourhood(i):
			neigh += (exp(- (BoW[i] - BoW[k])^2)/sum) 
		x[superpixel(i)] = (1-alpha) * x + alpha * (neigh)

img = mpimg.imread('../JPEGImages/2500_1.jpg')
img = cv2.resize(img,(100,150))

LAB_img = get_LAB(img)
hog_img = get_HOG(img)

sp_indices = np.array(mpimg.imread('2500_1.png'))

# To rescale tthe input between values 0 to 499
sp_indices = np.ceil((sp_indices/np.max(sp_indices)) * 499)
sp_indices = cv2.resize(sp_indices,(150,100))

BoW = np.array([None] * 500)

for i in range(500):
	req_indices = (np.isin(sp_indices,[i])).ravel()
	
	RGB_fts = np.array([])
	LAB_fts = np.array([])
	
	for j in range(3):
		curr = img[:,:,j].ravel()
		RGB_fts = np.append(RGB_fts,curr[req_indices])
	for j in range(3):
		curr = LAB_img[:,:,j].ravel()
		LAB_fts = np.append(LAB_fts,curr[req_indices])

	hog_img = hog_img.ravel()
	HOG_fts = hog_img[req_indices]

	BoW[i] = np.append(np.append(RGB_fts,LAB_fts),HOG_fts)

	f = open((str(i) + ".txt"), "w")

	for j in range(BoW[i].shape[0]):
		print(BoW[i].shape)
		f.write(str(BoW[i][j]) + " ")
	f.close()

x = cross_neighbourhood_voting(x)