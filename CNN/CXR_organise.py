# Code to extract and organize images for CNN
# Give it a csv of the data and how it's organized, along with 

# Pandas and Numpy are amazing for data organization
import pandas as pd
import numpy as np
# Tarfile and zipfile for extraction of the compressed files
import tarfile
import zipfile
# Some OS stuff for handling files and folders
import os
import shutil
import ntpath
import subprocess

# Declaring variables
csv_file = "Data_Entry_2017.csv"						# CSV file for how to organise the image data (which categories they belong to)
zipped_files = "archives"								# Where the raw image data is located
category_list = []											# Initializing a list of categories. Will be filled later
subfolder = "images"									# If the images are stored within a subfolder, write name in quotes. Otherwise use None
subfolder_length = len(subfolder) + 1

# This stuff you just have to know
image_name_column = "Image Index"						# Name of column in which "image name" is placed
category_column = "Finding Labels"						# Name of column in which "diagnosis" is placed
half_extract_command = None								# Declaring extract command. Depends on compression method

# Make data folders
shutil.rmtree('data', ignore_errors=True)
shutil.rmtree('tmp', ignore_errors=True)
os.mkdir('data')
os.mkdir('data/train')
os.mkdir('data/validation')
os.mkdir('data/test')
os.mkdir('tmp')

# Load the CSV file into variable "dataframe" so we can organise
dataframe = pd.read_csv(csv_file)
# There are labels that have multiple "diagnoses". Remove all but the first label. 
for index, row in dataframe.iterrows():
	if "|" in row[category_column]:
		dataframe.loc[index, category_column] = row[category_column].split('|')[0]


# Find all unique values in column "Finding labels". Basically a list of all diagnoses. Make a folder for each
category_list = dataframe[category_column].unique()
for category in category_list:
	if not os.path.isdir('data/train/' + str(category)):
		print "Making " + category + " directory"
		os.mkdir('data/train/' + str(category))
		os.mkdir('data/validation/' + str(category))
		os.mkdir('data/test/' + str(category))



# This isn't done yet
def organise():

	for image in os.listdir('tmp/images'):
		image_name = str(image)
		df_location = dataframe.loc[dataframe[image_name_column] == image_name]
		category = str(df_location[category_column])
		print image_name
		print df_location[category_column]
		subprocess.Popen("mv tmp/images/" + image_name + "data/train/" + category)



# Iterate through all files in your "tar_files" folder 
# Filename is the zipped file, and is extracted to 'tmp' in a folder with its own name
# Works for .tar, .tar.gz, .tgz, .tar.bz2, .tbz files.
def extract():
	print "========================"
	print "Extracting " + str(filename)
	for filename in os.listdir(zipped_files):
		os.mkdir('tmp/' + str(filename))
		# Firstly, load the zipped file into a temporary file... called file.
		if filename.endswith('.tar'):
			subprocess.Popen("tar xf " + filename + " tmp/")

		elif filename.endswith('.tar.gz') or filename.endswith('.tgz'):
			subprocess.Popen("tar xzf " + filename + " tmp/")

		elif filename.endswith('.tar.bz2') or filename.endswith('.tbz'):
			subprocess.Popen("tar xjf " + filename + " tmp/")


	# Now that we have the compressed file loaded, call the extract + organise functions
	print "Beginning extraction and organisation of images. Go grab a drink or somfin"
	extract()
	organise()
