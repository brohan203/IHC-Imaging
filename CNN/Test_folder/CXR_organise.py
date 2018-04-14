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

# Declaring variables
csv_file = "Data_Entry_2017.csv"						# CSV file for how to organise the image data (which categories they belong to)
zipped_files = "archives"								# Where the raw image data is located
category_list = []											# Initializing a list of categories. Will be filled later
subfolder = "images"									# If the images are stored within a subfolder, write name in quotes. Otherwise use None
subfolder_length = len(subfolder) + 1

# This stuff you just have to know
image_name_column = "Image Index"						# Name of column in which "image name" is placed
category_column = "Finding Labels"						# Name of column in which "diagnosis" is placed

# Make data folders
shutil.rmtree('data', ignore_errors=True)
os.mkdir('data')
os.mkdir('data/train')
os.mkdir('data/validation')
os.mkdir('data/test')

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
def extract_organise(file):
	for member in file.getmembers():
		#save = random.randint(1,3)
		# Get the file name
		_ , tail = ntpath.split(member.name)
		# Find the row in with the file's data in the CSV
		row = dataframe.where[dataframe[image_name_column] == tail]
		category = row[category_column]
		if not os.path.isdir("data/" + str(category)):
			os.mkdir("data/" + str(category))
			print "Created directory for " + category + "samples"
		file.extract(member, path='data/')


# Iterate through all files in your "tar_files" folder 
# Filename is the zipped file, and file is python's temporary loaded name for the file
# Works for .tar, .tar.gz, .tgz, .tar.bz2, .tbz, and zip files. Phew!
for filename in os.listdir(zipped_files):

	# Firstly, load the zipped file into a temporary file... called file.
	if filename.endswith('.tar'):
		file = tarfile.open(zipped_files + "/" + filename, 'r')

	elif filename.endswith('.tar.gz') or filename.endswith('.tgz'):
		file = tarfile.open(zipped_files + "/" + filename, 'r:gz')

	elif filename.endswith('.tar.bz2') or filename.endswith('.tbz'):
		file = tarfile.open(zipped_files + "/" + filename, 'r:bz2')

	elif filename.endswith('.zip'):
		file = zipfile.ZipFile(zipped_files + "/" + filename, 'r')



	print "========================"
	print "Extracting " + str(filename)

	# Now that we have the compressed file loaded, call the extract + organise function
	extract_organise(file)
