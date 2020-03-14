# Learn image
# Rohan Borkar, 2020

# Dependencies: PIL, matplotlib, skimage, sklearn, numpy
from PIL import Image, ImageOps, ImageChops
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.misc import imsave
from sklearn import svm
from sklearn.preprocessing import scale
from skimage import exposure, io
from skimage.color import rgb2lab
from skimage.morphology import watershed, remove_small_objects
from skimage.segmentation import mark_boundaries, slic
from skimage.measure import label, regionprops
from skimage.feature import structure_tensor, structure_tensor_eigvals
import os
from math import pi
import numpy as np
from sys import argv
import pickle
import cv2


class LearnImage:
	def __init__(self, image):
		# Ask how many clusters to segment the image into. Cluster = actual data, digit = cell type
		self.clusters = []
		self.digits = []

		# Load image (npimg), and record width, height, directory and image name
		self.image_name = str(image)
		self.np_image = io.imread(image).astype(np.uint8)
		self.lab_image = cv2.cvtColor(self.np_image, cv2.COLOR_RGB2LAB)
		self.gray_image = cv2.cvtColor(self.lab_image, cv2.COLOR_RGB2GRAY)
		# Separate R, G, B values. This leaves out A
		#self.r = exposure.equalize_adapthist(self.np_image[:,:,0], clip_limit=0.01)
		#self.g = exposure.equalize_adapthist(self.np_image[:,:,1], clip_limit=0.01)
		#self.b = exposure.equalize_adapthist(self.np_image[:,:,2], clip_limit=0.01)
		#self.np_image = np.dstack( (self.r, self.g, self.b) )
		self.r = self.np_image[:,:,0]
		self.g = self.np_image[:,:,1]
		self.b = self.np_image[:,:,2]

		self.image_width = self.np_image.shape[0]
		self.image_height = self.np_image.shape[1]
		self.head, self.tail = os.path.split(image)

		# Structure tensor texture arrays
		Axx, Axy, Ayy = structure_tensor(self.r, sigma=0.1)
		r_tensor = structure_tensor_eigvals(Axx, Axy, Ayy)
		Axx, Axy, Ayy = structure_tensor(self.g, sigma=0.1)
		g_tensor = structure_tensor_eigvals(Axx, Axy, Ayy)
		Axx, Axy, Ayy = structure_tensor(self.b, sigma=0.1)
		b_tensor = structure_tensor_eigvals(Axx, Axy, Ayy)
		tensor = np.stack( (r_tensor[0], g_tensor[0], b_tensor[0]), axis=2 )
		self.background = np.sum(self.gray_image > 150)

		fraction = float(self.background) / (self.image_width * self.image_height)

		# Aggregate all the R/G/B, LAB, and texture arrays together. Each time a point is selected, the script will record 9 data values for each pixel
		self.features = np.concatenate((self.np_image, self.lab_image, np.reshape(self.gray_image, (self.image_width, self.image_height, 1)), tensor), axis=2)

	# Select points comes from my old script, it allows the user to right click on points on the image to record them as training data
	# NEVER call this script, it is called by "collect_data"
	def select_points(self, npimage, features, cluster):
		diameter = 2 # Diameter of circle to draw and record when points are selected
		diff = int(diameter/2) # Radius
		divide = (2*int(diameter/2))**2 # What to divide by when adding to r/g/b lists
		# Starting to implement "lab" colour space
		fig1 = plt.figure()
		ax1 = fig1.add_subplot(111)
		ax1.imshow(npimage, cmap=plt.cm.gray)
		plt.title("Select at least ten points for training data " + str(cluster+1))

	    #function to be called when mouse is clicked
		def onclick(event):
			if event.button == 3:
				ix, iy = int(event.xdata), int(event.ydata)
				ax1.add_patch(patches.Circle((ix, iy), radius=diff, color='black'))
				for a in range(0-diff, diff):
						for b in range(0-diff, diff):
							# Add surrounding pixel data along with specific label
							self.clusters.append( features[iy+b, ix+a] )
							self.digits.append(cluster)
				fig1.canvas.draw()
		cid = fig1.canvas.mpl_connect('button_release_event', onclick)
		plt.show()

	# Calls select_points for each cluster the user has specified. Collects training data.
	def collect_data(self, image):# Either collect training data\
		self.select_points(self.np_image, self.features, 0)
		self.select_points(self.np_image, self.features, 1)
	
	# Training the model
	def model(self):
		# Arrange training data into np arrays
		self.clusters = np.asarray(self.clusters)
		self.digits = np.asarray(self.digits)
		# Train (fit) the model on user input data
		self.clf = svm.SVR(gamma=0.001, C=100.)
		self.clf.fit(self.clusters, self.digits)

	# Predict the entire image based on the model 
	def analyze_image(self):
		feats = np.reshape(self.features, ((self.image_width*self.image_height), 10))
		self.predicted_img = self.clf.predict(feats)
		self.predicted_img = np.array(self.predicted_img)
		self.predicted_img = np.reshape(self.predicted_img, (self.image_width, self.image_height))
		self.predicted_img[self.predicted_img > 0.5] = 1
		self.predicted_img[self.predicted_img <= 0.5] = 0

		segment_area = np.sum(self.predicted_img == 1.0)
		tissue_area = (self.image_width * self.image_height) - float(self.background)
		percent_segment = float(segment_area) / float(tissue_area)
		print (percent_segment)

		#Make a plot to represent the image
		fig, ax = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True, subplot_kw={'adjustable': 'box-forced'})
		ax1, ax2 = ax.ravel()
		ax1.imshow(self.np_image, cmap=plt.cm.gray)
		ax2.imshow(self.predicted_img, cmap=plt.cm.gray)
		plt.show()

	# Trying out watershed in order to separate connecting cells 
	def sort_labels(self):
		self.labels = label(self.predicted_img)
		self.labels = remove_small_objects(self.labels, 10)
		# Sort normal, too-big, and too-small objects

		# Visualize
		# fig, ax = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True, subplot_kw={'adjustable': 'box-forced'})
		# ax1, ax2 = ax.ravel()
		# ax1.imshow(self.np_image)
		# ax2.imshow(self.labels)
		# plt.show()

	def load_data(self):
		self.clusters = pickle.load( open( "clusters_green2.p", "rb" ) )
		self.digits = pickle.load( open(  "digits_green2.p", "rb" ) )


	# Save an image.
	def save_image(self, image, name):
		# Available images:
			# Masked image = self.mask
			# Binary image = self.predicted_img
			# Original image = self.np_image
		imsave(name, image)

	# Save or load training data. Put this in the "run" function at the bottom of the script
	def save_trainingdata(self, image):
		pickle.dump( self.clusters, open( "clusters" + image + ".p", "wb" ) )
		pickle.dump( self.digits, open( "digits" + image + ".p", "wb" ) )

	def run(self, image):

		self.collect_data(image)
		#self.save_trainingdata(image)
		#self.load_data()
		self.model()
		self.analyze_image()
		self.sort_labels()
		self.save_image(self.predicted_img, ('output.png'))
		self.save_image( mark_boundaries(self.np_image, self.labels), ('boundaries.png'))

# Run a batch of images in folder IBA-1gfap
# image_folder = 'IBA-1gfap'
# for folder in os.listdir(image_folder):
# 	if folder[0:3] == 'igr' or folder[0:3] == 'igl':
# 		print (folder)
# 		for subfolder in os.listdir(image_folder + '/' + folder):
# 			print (subfolder)
# 			for image in os.listdir(image_folder + '/' + folder + '/' + subfolder):
# 				if image == "ROI.tif":
# 					img = LearnImage(image_folder + '/' + folder + '/' + subfolder + '/' + image)
# 					img.run(image_folder, folder, subfolder)

# Replace "ROI.tif" to whatever file you'd like to analyze
img = LearnImage('ROI.tif')
img.run('ROI.tif')
