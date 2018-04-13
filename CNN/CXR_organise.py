# Code to extract and organize images for CNN
# Give it a csv of the data and how it's organized, along with 


import pandas as pd
import numpy as np
import tarfile
import zipfile
import os
import shutil

csv_file = "Data_Entry_2017.csv"
zipped_files = "archives"
categories = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

image_name_column = 1
category_column = 2


pd.read_csv(csv_file)

def extract_organise(file):
	



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

	# Now we have a compressed file. Start extracting using the 
