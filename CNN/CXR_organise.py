# Code to extract and organize images for CNN
# Give it a csv of the data and how it's organized, along with 


import pandas as pd
import numpy as np
import tarfile
import zipfile
import os
import shutil

# Declaring variables
csv_file = "Data_Entry_2017.csv"						# CSV file for how to organise the image data (which categories they belong to)
zipped_files = "/archives"								# Where the raw image data is located
categories = []											# Initializing a list of categories. Will be filled later

# This stuff you just have to know
image_name_column = "Image_name"						# Name of column in which "image name" is placed
category_column = "Diagnosis"							# Name of column in which "diagnosis" is placed


dataframe = pd.read_csv(csv_file)


# This isn't done yet
def extract_organise(file):
	for member in file.getmembers()
		row = dataframe.loc[dataframe[image_name_column] == str(member)]
		file.extract(member, '/data/' + str(dataframe[category_column[row]]))



shutil.rmtree('/data', ignore_errors=True)
os.mkdir('/data')
os.mkdir('/data/train')
os.mkdir('/data/validation')
os.mkdir('/data/test')

# Iterate through all files in your "tar_files" folder 
# Filename is the zipped file, and file is python's temporary loaded name for the file
# Works for .tar, .tar.gz, .tgz, .tar.bz2, .tbz, and zip files. Phew!
for filename in os.listdir(zipped_files):

	# Firstly, load the zipped file into a temporary file... called file.
	if filename.endswith('.tar'):
		file = tarfile.open(filename, 'r')

	elif filename.endswith('.tar.gz') or filename.endswith('.tgz'):
		file = tarfile.open(filename, 'r:gz')

	elif filename.endswith('.tar.bz2') or filename.endswith('.tbz'):
		file = tarfile.open(filename, 'r:bz2')

	elif filename.endswith('.zip'):
		file = zipfile.ZipFile(filename, 'r')

	# Now that we have the compressed file loaded, call the extract + organise function
	extract_organise(file, )
