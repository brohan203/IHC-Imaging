# Learn image
# Rohan Borkar, May 2016

# Dependencies: PIL, matplotlib, skimage, sklearn, numpy
from PIL import Image, ImageOps, ImageChops
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import ndimage as ndi
from collections import defaultdict
from sklearn import svm
from skimage.color import rgb2lab
from skimage.filters import gaussian, sobel
from skimage.filters.rank import gradient
from skimage.morphology import watershed, disk
from skimage.segmentation import mark_boundaries, slic
from skimage.measure import label, regionprops
from skimage.feature import structure_tensor, structure_tensor_eigvals, peak_local_max
import pandas as pd
import os, math
import numpy as np
from sys import argv

script, image = argv

class LearnImage:
	def __init__(self, image):
		# Ask how many clusters to segment the image into
		self.clusters = []
		self.digits = []

		# Load image (npimg), and record width, height, directory and image name
		self.image_name = str(image)
		pilimg = Image.open(image)
		self.np_image = np.asarray(pilimg)
		print self.np_image.shape

		self.image_width = self.np_image.shape[0]
		self.image_height = self.np_image.shape[1]
		self.head, self.tail = os.path.split(image)
		self.sobel_image = sobel(self.np_image)
		self.sobel_image.setflags(write=True)

		print "Finished importing and processing image"
 
	# Select points comes from my old script, it allows the user to right click on points on the image to record them as training data
	# NEVER call this script, it is called by "collect_data"
	def select_points(self, npimage, cluster):
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
							self.clusters.append( npimage[iy+b, ix+a] )
							self.digits.append(cluster)
		cid = fig1.canvas.mpl_connect('button_release_event', onclick)
		plt.show()

	# Calls select_points for each cluster the user has specified. Collects training data.
	def collect_data(self):# Either collect training data\
		if self.pickle_data == False:
			self.segs = input("Enter number of clusters: ")
			for x in range(0, self.segs):
				self.select_points(self.np_image, x)
			print "Finished collecting training data"
		else:
			self.clusters = pickle.load( open( self.image_name + "clusters.p", "rb" ) )
			self.digits = pickle.load( open( self.image_name + "digits.p", "rb" ) )

	# Training the model
	def model(self):
		print "Fitting data..."
		# Arrange training data into np arrays
		self.clusters = np.asarray(self.clusters)
		self.digits = np.asarray(self.digits)
		# Train (fit) the model on user input data
		self.clf = svm.SVC(gamma=0.001, C=100.)
		self.clf.fit(self.clusters, self.digits)

	# Predict the entire image based on the model 
	def analyze_image(self):
		print "Predicting image..."
		self.features = np.reshape(self.np_image, ((self.image_width*self.image_height), 2))
		self.predicted_img = self.clf.predict(self.features)
		self.predicted_img = np.array(self.predicted_img)
		self.predicted_img = np.reshape(self.predicted_img, (self.image_width, self.image_height))
		self.features = np.reshape(self.predicted_img, (self.image_width, self.image_height))

		# Make a plot to represent the image
		fig, ax = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True, subplot_kw={'adjustable': 'box-forced'})
		ax1, ax2 = ax.ravel()
		ax1.imshow(self.np_image, cmap=plt.cm.gray)
		ax2.imshow(self.predicted_img, cmap=plt.cm.gray)
		plt.show()


	# Save an image.
	def save_image(self, image):
		# Available images:
			# Masked image = self.mask
			# Binary image = self.predicted_img
			# Original image = self.np_image
		if image:
			plt.imsave(fname=(str(image) + self.tail), arr=image)
			print "Image has been saved."

	# Save or load training data. Put this in the "run" function at the bottom of the script
	def save_trainingdata(self):
		if self.pickle_data == False:
			pickle.dump( self.clusters, open( self.image_name + "clusters.p", "wb" ) )
			pickle.dump( self.digits, open( self.image_name + "digits.p", "wb" ) )

	def run(self):
		self.pickle_data = False
		self.collect_data()
		self.model()
		self.analyze_image()
		self.watershed()

img = LearnImage(image)
img.run()





