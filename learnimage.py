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
		# Separate R, G, B values. This leaves out A
		r = self.np_image[:,:,0]
		g = self.np_image[:,:,1]
		b = self.np_image[:,:,2]
		self.np_image = np.dstack( (r, g, b) )
		pilimg = pilimg.convert('L')
		self.grayscale_image = np.asarray(pilimg)
		self.grayscale_image.setflags(write=True)
		self.image_width = self.np_image.shape[0]
		self.image_height = self.np_image.shape[1]
		self.head, self.tail = os.path.split(image)
		self.gaussian_image = gaussian(self.np_image, 2, multichannel=True)
		self.gaussian_image.setflags(write=True)

		# Structure tensor texture arrays
		Axx, Axy, Ayy = structure_tensor(r, sigma=0.1)
		r_tensor = structure_tensor_eigvals(Axx, Axy, Ayy)
		Axx, Axy, Ayy = structure_tensor(g, sigma=0.1)
		g_tensor = structure_tensor_eigvals(Axx, Axy, Ayy)
		Axx, Axy, Ayy = structure_tensor(b, sigma=0.1)
		b_tensor = structure_tensor_eigvals(Axx, Axy, Ayy)
		tensor = np.stack( (r_tensor[0], g_tensor[0], b_tensor[0]), axis=2 )
		# Aggregate all the R/G/B, LAB, and texture arrays together. Each time a point is selected, the script will record 9 data values for each pixel
		self.features = np.concatenate((self.gaussian_image, rgb2lab(self.np_image), tensor), axis=2)
		print "Finished importing and processing image"
  #   def extract_gabor(self):
  #   	lab = color.rgb2lab(self.image)
		# lab1 = lab[:,:,0]
		# lab2 = lab[:,:,1]
 	# 	lab3 = lab[:,:,2]
	 #                ## extract lab intensity space corresponding to nuclei
  #       # nuclei = lab3
  #       im_grey = filters.gaussian(lab3, 1)
  #       ## flatten lab channels
  #       lab1 = np.reshape(lab[:,:,0], (1, self.dimensions))
  #       lab2 = np.reshape(lab[:,:,1], (1, self.dimensions))
  #       lab3 = np.reshape(lab[:,:,2], (1, self.dimensions))
  #       ## stack lab channels into one array
  #       lab_features = np.vstack((scale(lab1, axis=1),scale(lab2, axis=1),scale(lab3, axis=1)))

  #       frequencies = [1.571]
  #       orientation_rads = [0, 0.785398, 1.5708, 2.35619]
  #       ## generate list of combinations
  #       combos = list(IT.product(frequencies, orientation_rads))
  #       list_of_mags = []
  #       for item in combos:
  #               real_gabor, imaginary_gabor = filters.gabor(im_grey, item[0], item[1])
  #               ## find the magnitude (square root of the sum of squares of real and imaginary gabor
  #               mag = np.sqrt(np.square(real_gabor, dtype=np.float64)+np.square(imaginary_gabor, dtype=np.float64))
  #               sigma = .5 * ((2*pi)/item[0])
  #               K = 2
  #               ## apply gaussian filter and scale the result with zero mean and unit variance
  #               mag_gaussian = filters.gaussian(mag, K*sigma)
  #               mag_gaussian_flattened = np.reshape(mag_gaussian, (1, (self.x*self.y)))
  #               list_of_mags.append(scale(mag_gaussian_flattened, axis=1))
  #       list_of_mags = np.asarray(list_of_mags).reshape(len(combos),self.dimensions)
  #       ## combine gabor features with lab features
  #       im_grey_flattened = np.reshape(im_grey, (1, (self.x*self.y)))
  #       features = np.vstack((im_grey_flattened, list_of_mags))
  #       return features

	# Select points comes from my old script, it allows the user to right click on points on the image to record them as training data
	# NEVER call this script, it is called by "collect_data"
	def select_points(self, npimage, rgblab_img, cluster):
		diameter = 12 # Diameter of circle to draw and record when points are selected
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
							self.clusters.append( rgblab_img[iy+b, ix+a] )
							self.digits.append(cluster)
		cid = fig1.canvas.mpl_connect('button_release_event', onclick)
		plt.show()

	# Calls select_points for each cluster the user has specified. Collects training data.
	def collect_data(self):# Either collect training data\
		if self.pickle_data == False:
			self.segs = input("Enter number of clusters: ")
			for x in range(0, self.segs):
				self.select_points(self.np_image, self.features, x)
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
		self.features = np.reshape(self.features, ((self.image_width*self.image_height), 9))
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

	def mask(self):
		print "Masking image..."
		# Labels takes the results of "predict image" and creates a separate object for each set of connected pixels
		labels = label(self.predicted_img)

		# Separate into r/g/b arrays for masking. This may not be the most optimal method
		r = self.np_image[:,:,0]
		g = self.np_image[:,:,1]
		b = self.np_image[:,:,2]
		r.setflags(write=True)
		g.setflags(write=True)
		b.setflags(write=True)
		r[labels == 0] = 0
		g[labels == 0] = 0
		b[labels == 0] = 0
		self.mask = np.stack( (r, g, b), axis=2 )

	# Trying out watershed in order to separate connecting cells 
	def watershed(self):
		min_cell_size = 20
		print "Labeling data..."
		# Small labels are objects that are small enough that they don't need to be further segmented
		smalllabels = label(self.predicted_img)
		measurements = regionprops(smalllabels)

		# Big labels are objects that are too big (probably multiple cells) and need to be further segmented
		biglabels = np.zeros( (smalllabels.shape) )
		# Record average perimeter of objects, along with standard deviation of perimeters
		avg_perimeter = sum((x.perimeter for x in measurements))/len(measurements)
		std = np.std(list(x.perimeter for x in measurements))
		count = 0
		# Sort normal, too-big, and too-small objects
		for region in regionprops(smalllabels):
			# # Too small
			if region.area < min_cell_size:
				smalllabels[smalllabels == region.label] = 0
				pass
			# Too big
			elif region.perimeter >= (1.5*std + avg_perimeter):
				count+= 1
				biglabels[smalllabels == region.label] = 1
				smalllabels[smalllabels == region.label] = 0

		# Watershed segmentation. I have no idea how this works. I'm a monkey with a typewriter
		# Only run watershed on BIG labels because those are assumed to be the only ones that need to be further segmented
		distance = ndi.distance_transform_edt(biglabels)
		local_maxi = peak_local_max(distance, indices=False, footprint=np.ones( (avg_perimeter/2,avg_perimeter/2) ), labels=biglabels)
		markers = label(local_maxi)
		biglabels = watershed(-distance, markers, mask=biglabels)

		# Add the original small objects with the new ones that were segmented out of bigger objects
		labels = label( (smalllabels + biglabels) )
		

		# Visualize
		fig, ax = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True, subplot_kw={'adjustable': 'box-forced'})
		ax1, ax2 = ax.ravel()
		ax1.imshow(self.np_image)
		ax2.imshow(labels)
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
		#self.save_trainingdata()
		self.model()
		self.analyze_image()
		self.watershed()

img = LearnImage(image)
img.run()





