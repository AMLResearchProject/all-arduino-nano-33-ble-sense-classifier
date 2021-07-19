#!/usr/bin/env python3
""" AI Model Data Class.

Provides the AI Model with the required required data
processing functionality.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files(the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Contributors:
- Adam Milton-Barker

"""

import cv2
import os
import pathlib

import numpy as np

from numpy.random import seed
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle

from modules.AbstractData import AbstractData
from modules.augmentation import augmentation

class data(AbstractData):
	""" AI Model Data Class.

	Provides the AI Model with the required required data
	processing functionality.
	"""

	def process(self):
		""" Processes the images. """

		aug = augmentation(self.helpers)

		data_dir = pathlib.Path(self.confs["data"]["train_dir"])
		data = list(data_dir.glob(
			'*' + self.confs["data"]["file_type"]))

		count = 0
		neg_count = 0
		pos_count = 0

		augmented_data = []
		self.labels = []
		temp = []

		for rimage in data:
			fpath = str(rimage)
			fname = os.path.basename(rimage)
			label = 0 if "_0" in fname else 1

			# Resize Image
			image = self.resize(fpath, self.dim)

			if image.shape[2] == 1:
				image = np.dstack(
					[image, image, image])

			temp.append(image.astype(np.float32)/255.)

			self.data.append(image.astype(np.float32)/255.)
			self.labels.append(label)

			# Grayscale
			self.data.append(aug.grayscale(image))
			self.labels.append(label)

			# Histogram Equalization
			self.data.append(aug.equalize_hist(image))
			self.labels.append(label)

			# Reflection
			horizontal, vertical = aug.reflection(image)
			self.data.append(horizontal)
			self.labels.append(label)
			self.data.append(vertical)
			self.labels.append(label)

			# Gaussian Blur
			self.data.append(aug.gaussian(image))
			self.labels.append(label)

			# Translation
			self.data.append(aug.translate(image))
			self.labels.append(label)

			# Shear
			self.data.append(aug.shear(image))
			self.labels.append(label)

			# Rotation
			for i in range(0, self.helpers.confs["data"]["rotations"]):
				self.data.append(aug.rotation(image))
				self.labels.append(label)
				if "_0" in fname:
					neg_count += 1
				else:
					pos_count += 1
				count += 1

			if "_0" in fname:
				neg_count += 8
			else:
				pos_count += 8
			count += 8

		self.shuffle()
		self.convert_data()
		self.encode_labels()

		self.helpers.logger.info("Augmented data size: " + str(count))
		self.helpers.logger.info("Negative data size: " + str(neg_count))
		self.helpers.logger.info("Positive data size: " + str(pos_count))
		self.helpers.logger.info("Augmented data shape: " + str(self.data.shape))
		self.helpers.logger.info("Labels shape: " + str(self.labels.shape))

		self.X_train_arr = np.asarray(temp)

		self.get_split()

	def convert_data(self):
		""" Converts the training data to a numpy array. """

		self.data = np.array(self.data)

	def encode_labels(self):
		""" One Hot Encodes the labels. """

		encoder = OneHotEncoder(categories='auto')

		self.labels = np.reshape(self.labels, (-1, 1))
		self.labels = encoder.fit_transform(self.labels).toarray()

	def shuffle(self):
		""" Shuffles the data and labels. """

		self.data, self.labels = shuffle(
			self.data, self.labels, random_state=self.seed)

	def get_split(self):
		""" Splits the data and labels creating training and validation datasets. """

		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
			self.data, self.labels, test_size=self.helpers.confs["data"]["split"],
			random_state=self.seed)

		self.helpers.logger.info("Training data: " + str(self.X_train.shape))
		self.helpers.logger.info("Training labels: " + str(self.y_train.shape))
		self.helpers.logger.info("Validation data: " + str(self.X_test.shape))
		self.helpers.logger.info("Validation labels: " + str(self.y_test.shape))

	def resize(self, path, dim):
		""" Resizes an image to the provided dimensions (dim). """

		return cv2.resize(cv2.imread(path), (dim, dim))
