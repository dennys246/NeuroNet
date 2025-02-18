from __future__ import absolute_import, division, print_function, unicode_literals
import os, csv, shutil, time, random, psutil
import numpy as np
import nibabel as nib
from glob import glob
from scipy.signal import fftconvolve
from scipy.stats import gamma

class wrangler:
	def __init__(self, config):
		self.config = config
		return
	
	def wrangle(self, subject_pool, subjects = [], count = 1, session = '*', fold_size = 10, shuffle = False, resolution = np.float16, activation = 'linear',  exclude_trained = False):
		# Wrangle subjects into training and validation batches, using the fold size
		# to determine how many subjects to put in the training set compared to validation
		#
		# NOTE: This function is not typically used for grabing testing subjects, consider
		# Using the load subject function directly for testing

		print(f"Wrangling {count} subjects (session  {session}) with fold of size {fold_size}.")

		print(f"Training portion: {round(((fold_size - 2)/fold_size)*100, 1)}%\nValidation portion: {round((1/fold_size)*100, 1)}%\nTesting portion: {round((1/fold_size)*100, 1)}%")
	
		# Define the subject count found
		subject_count = 0
		if count == 0 and len(subjects) == 0: # Check that enough subjects were passed in
			print("Not enough subjects passed into wrangle, must pass in at least 1 subject or a count of 1 specified...")
			return 
		
		# Run through each subject and load data
		self.training_batch = []
		
		# Define variables to hold subject data
		images = np.array([])
		labels = np.array([])

		train_indices = []
		val_indices = []

		# While we haven't reached out subject count goal (minimum of 2)
		while subject_count < count:

			# If we've run out of subjects but need more
			if subjects == []: 
				subject = self.config.subject_pool.pop(random.randint(0, len(self.config.subject_pool) - 1))
			else: # Grab the first subject in our passes subject list
				subject = subjects.pop(0)
			print(subject)

			# Check if subject is in excluded list
			if subject in self.config.excluded_pool: 
				print(f'Excluding subject {subject} from run...')
				continue

			# If subject previously run and we're not retraining\
			if exclude_trained == True and sum([subject == trained_subject for trained_subject in self.config.trained_pool]) > 0: # check is subject previously run
				print(f"Subject {subject} previously run, skipping subject...")
				continue

			# Assess subject position in fold
			viewed_count = len(self.config.trained_pool) + len(self.config.validation_pool) + len(self.config.test_pool) + len(self.training_batch)
			position = viewed_count % fold_size

			if position < (fold_size - 1): # If not a test subject at the end of a fold
				image, label = self.load_subject(subject, session, shuffle, resolution, activation)
				if len(label) == 0: # If no images available
					print(f"Skipping {subject}, no labels...")
					continue # Skip subjects
				if position != 0: # Append images
					try: # appending the images to the image object
						images = np.append(images, image, axis = 0)
						labels = np.append(labels, label)
					except: # initialize a new image object
						images = image
						labels = label

			# Divide up data for training, testing and validating to achieve 80/10/10
			
			# Move through the fold
			if position == 0: # Grab the validation image
				self.validation_image = image
				self.validation_labels = label
				self.config.validation_pool.append(subject)
				continue # Continue without incrementing subject count

			elif position == (fold_size - 1): # Grab 
				self.config.test_pool.append(subject)

			else: # Grab training and validation indices indices
				train_indices += [ind for ind in range(len(train_indices), len(train_indices) + len(image) - 1 )]

				val_start = int(round(self.validation_image.shape[0] * ((position-1)/8), 0))

				val_end = int(round(self.validation_image.shape[0] * (position/8), 0))
				if val_end > self.validation_image.shape[0]: val_end = self.validation_image.shape[0]

				val_indices += [ind for ind in range(val_start, val_end)]

				self.training_batch.append(subject)

			subject_count += 1

		# Check if images loaded had any volumes in then
		if images.shape[0] == 0:
			return False
		
		print(f'Train indices count: {len(train_indices)}\nValidation indices count: {len(val_indices)}\nLength of labels: {len(labels)}')
		x_train = images[train_indices,:,:,:]
		y_train = labels[train_indices]
		x_val = self.validation_image[val_indices,:,:,:]
		y_val = self.validation_labels[val_indices]

		return x_train, y_train, x_val, y_val


	def load_subject(self, subject, session, shuffle = False, resolution = np.float16, activation = 'linear',  load_affine = False, load_header = False):
		
		print(f"Loading subject {subject}")

        # Create a call for handling an empty empty exit <-- Update with **kwargs
		def empty_exit(load_affine, load_header):
			if load_affine == False and load_header == False:
				return [], []
			if load_header == False and load_affine == True:
				return [], [], []
			if load_header == True and load_affine == False:
				return [], [], []
			if load_header == True and load_affine == True:
				return [], [], [], []
	
		image_file = self.load_image(subject, session, memory_map = True)
	
		# Grab images header/meta data
		header = image_file.header 
		self.config.header = header

		affine = image_file.affine

        # Grab data
		image = np.array(image_file.get_fdata(dtype=resolution))

		# Reshape image to have time dimension as first dimension and add channel dimension
		image = image.reshape((image.shape[3], image.shape[0], image.shape[1], image.shape[2], 1))
		self.config.data_shape = image.shape[1:-1]
		print(f"Subject {subject} data shape: {image.shape}")

        # Normalize data
		image = self.normalize(image)

		# Load labels
		labels = self.load_labels(subject, session)
		print(f'Subject {subject} label shape: {labels.shape}')
        
        # If no labels ended up being loaded, exit
		if labels.shape[0] == 0 or labels.shape[0] != image.shape[0]:
			print(f"Labels are empty or labels length does not match image length...\n Image shape - {image.shape}\n Label shape - {labels.shape})")
			return empty_exit(load_affine, load_header)
		
		if activation != 'linear': # If not a regression problem
			labels = [int(label) for label in labels] # Convert labels to integers for classifing
        
        #  Shuffle images and labels if configured
		if self.config.shuffle == True or shuffle == True:
			image, labels = self.shuffle(image, labels)

        # Return requested subject data <-- Update with **kwargs
		if load_affine == False and load_header == False:
			return image, labels
		if load_header == False and load_affine == True:
			return image, labels, affine
		if load_header == True and load_affine == False:
			return image, labels, header
		if load_header == True and load_affine == True:
			return image, labels, header, affine
		
	def load_image(self, subject, session, memory_map = False):
		# look for all relavent image file
		image_filenames = glob(f"{self.config.bids_directory}/derivatives/{self.config.tool}/{subject}/ses-{session}/func/{self.config.bold_identifier}")
		if image_filenames: #  if no image files found, return empty handed
			if len(image_filenames) > 1:
				print(f"WARNING: Multiple BOLD files found for subject {subject} session {session} using BOLD identifier {self.config.bold_identifier} provided, using first file found which could cause instability later in NeuroNet...")
			print(f"Loading BOLD image {image_filenames[0]}")
			return nib.load(image_filenames[0], mmap=memory_map) # Load images
		else:
			print(f'No images found for {subject} ses-{session} using BOLD identifier {self.config.bold_identifier}')
			return False
	
	def load_labels(self, subject, session):
		# Grab relavent label filenames
		label_filenames = glob(f"{self.config.bids_directory}/derivatives/{self.config.tool}/{subject}/ses-{session}/func/{self.config.label_identifier}")
		if label_filenames: # If we grabbed more/less than one label file, exit
			if len(label_filenames) > 1:
				print(f"WARNING: Mutliple labels found for {subject} session {session}, grabbing first label file found however this may cause instability in NeuroNet...")
			label_filename = label_filenames[0]
		else: # Alert of no labels found...
			label_filenames = '\n'.join(label_filenames)
			print(f"No label files found for subject {subject} session {session}:\n {label_filenames}")
			return False
		
		# Check if correct format
		#if label_filename[-4:] not in ['.txt', '.csv', '.tsv']:
		#	print(f"Error: file not in correct format, adjust file format to be .txt, .csv or .tsv. If file was not intended to be read, check that your label identifier {self.config.label_identifier} is unique to the file you want to load in.\n\nAttempted file loaded: {label_filename}")
			
		print(f"Loading label file {label_filename}")
		# initialize empty labels and read in labels
		labels = []
		with open(label_filename, 'r') as label_file:
            # Handle text file case
			if label_filename[-4:] == '.txt': 
				labels = label_file.readlines()
				labels = [float(label) for label in ''.join(labels).split('\n') if label != '']
            
            # Handle csv file case
			if label_filename[-4:] == '.csv':
				labels = []
				csv_reader = csv.reader(label_filename)
				for row in csv_reader:
					labels.append(float(row))

			# Handle tsv file case
			if label_filename[-4:] == '.tsv':
				tsv_reader = csv.reader(label_filename, '\t')
				for row in tsv_reader:
					labels.append(float(row))

		return np.array(labels)

	def load_shape(self, subject, session, image = None):
		if image == None:
			image = self.load_image(subject, session)
		return image.header.get_data_shape()

	def shuffle(self, images, labels):
		# Get sample indices
		indices = np.arange(images.shape[0])
		
        # Shuffle indices
		np.random.shuffle(indices)

		# Rearrange images and labels using shuffled indices
		images = images[indices, :, :, :, :]
		
		labels = labels[indices]
		
		return images, labels
	
	def normalize(self, array):
		# Find array min
		array_min = np.min(array)

		# Find array max
		array_max = np.max(array)

		# Pass back normalized array
		return 2 * (array - array_min) / (array_max - array_min) - 1


	def mean_filter(self, image, filter_size):
		return
	
	def median_filter(self, image, filter_size):
		return

	def gaussian_filter(self, image, filter_size, sigma = 1):
		return
	
	def bilateral_filter(self, image, filter_size):
		return
	
	def laplacian_filter(self, image, filter_size, sign = 'positive'):
		return

	def dilation_filter(self, image, filter_size):
		return

	def erosion_filter(self, image, filter_size):
		return

	def display_memory(self, title):
		# Get the memory information
		memory = psutil.virtual_memory()
		swap = psutil.swap_memory()

		print(f"|------ {title} -------|")

		# Display RAM usage
		print("---- System Memory (RAM) ----")
		print(f"Total RAM: {memory.total / (1024 ** 3):.2f} GB")
		print(f"Available RAM: {memory.available / (1024 ** 3):.2f} GB")
		print(f"Used RAM: {memory.used / (1024 ** 3):.2f} GB")
		print(f"RAM Usage: {memory.percent}%")

		# Display swap memory usage
		print("\n---- Swap Memory ----")
		print(f"Total Swap: {swap.total / (1024 ** 3):.2f} GB")
		print(f"Used Swap: {swap.used / (1024 ** 3):.2f} GB")
		print(f"Free Swap: {swap.free / (1024 ** 3):.2f} GB")
		print(f"Swap Usage: {swap.percent}%")


	def create_dir(self):
		print(self.config.outputs_category)
		first_dir = self.config.outputs_category # Create lists of all directory levels for extraction outputs
		second_dir = [f'Layer_{str(layer)}' for layer in range(1, self.config.convolution_depth)]
		third_dir = ["DeConv_Feature_Maps", "DeConv_CAM"]
		fourth_dir = ["GB", "SM", "SSM"]
		if self.config.rebuild == True:
			if os.path.exists(f'{self.config.project_directory}{self.config.model_directory}/') == True:
				print(f'\nRun directory {self.config.project_directory}{self.config.model_directory} already exists, clearing directory...')
				shutil.rmtree(f'{self.config.project_directory}{self.config.model_directory}')
				time.sleep(1)
		else: # If not resetting model
			if os.path.isdir(f'{self.config.project_directory}{self.config.model_directory}/') == True: # If model exists
				print(f"Run directory already exists for {self.config.model_directory}, if you intended to train a fresh model consider altering run directory or deleting the model saved in {self.config.model_directory} (i.e. setting config.reset_model = True)")
				return

		os.mkdir(f'{self.config.project_directory}{self.config.model_directory}/')
		os.mkdir(f'{self.config.project_directory}{self.config.model_directory}/model')
		os.mkdir(f'{self.config.project_directory}{self.config.model_directory}/plots')
		os.mkdir(f'{self.config.project_directory}{self.config.model_directory}/jack_knife')
		os.mkdir(f'{self.config.project_directory}{self.config.model_directory}/jack_knife/probabilities')
		for first in first_dir:
			os.mkdir(f'{self.config.project_directory}{self.config.model_directory}/{first}')
			for second in second_dir:
				os.mkdir(f'{self.config.project_directory}{self.config.model_directory}/{first}/{second}')
				for third in third_dir:
					os.mkdir(f'{self.config.project_directory}{self.config.model_directory}/{first}/{second}/{third}')
					for fourth in fourth_dir:
						os.mkdir(f'{self.config.project_directory}{self.config.model_directory}/{first}/{second}/{third}/{fourth}')
		print(f'\nResult directories generated for {self.config.model_directory}\n')
