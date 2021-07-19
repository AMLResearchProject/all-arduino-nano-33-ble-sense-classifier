#!/usr/bin/env python3
""" Data Augmentation Class.

Provides data augmentation methods.

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
import random

import numpy as np

from numpy.random import seed
from scipy import ndimage
from skimage import transform as tm

class augmentation():
	""" HIAS AI Model Data Augmentation Class

	Provides data augmentation methods.
	"""

	def __init__(self, helpers):
		""" Initializes the class. """

		self.helpers = helpers

		self.seed = self.helpers.confs["data"]["seed"]
		seed(self.seed)

		self.helpers.logger.info("Augmentation class initialization complete.")

	def grayscale(self, data):
		""" Creates a grayscale copy. """

		gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
		return np.dstack([gray, gray, gray]).astype(np.float32)/255.

	def equalize_hist(self, data):
		""" Creates a histogram equalized copy. """

		img_to_yuv = cv2.cvtColor(data, cv2.COLOR_BGR2YUV)
		img_to_yuv[:, :, 0] = cv2.equalizeHist(img_to_yuv[:, :, 0])
		hist_equalization_result = cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2BGR)
		return hist_equalization_result.astype(np.float32)/255.

	def reflection(self, data):
		""" Creates a reflected copy. """

		return cv2.flip(data, 0).astype(np.float32)/255., cv2.flip(data, 1).astype(np.float32)/255.

	def gaussian(self, data):
		""" Creates a gaussian blurred copy. """

		return ndimage.gaussian_filter(data, sigma=5.11).astype(np.float32)/255.

	def translate(self, data):
		""" Creates transformed copy. """

		cols, rows, chs = data.shape

		return cv2.warpAffine(data, np.float32([[1, 0, 84], [0, 1, 56]]), (rows, cols),
							  borderMode=cv2.BORDER_CONSTANT, borderValue=(144, 159, 162)).astype(np.float32)/255.

	def rotation(self, data):
		""" Creates a rotated copy. """

		cols, rows, chs = data.shape

		rand_deg = random.randint(-180, 180)
		matrix = cv2.getRotationMatrix2D((cols/2, rows/2), rand_deg, 0.70)
		rotated = cv2.warpAffine(data, matrix, (rows, cols), borderMode=cv2.BORDER_CONSTANT,
								borderValue=(144, 159, 162))

		return rotated.astype(np.float32)/255.

	def shear(self, data):
		""" Creates a histogram equalized copy. """

		at = tm.AffineTransform(shear=0.5)
		return tm.warp(data, inverse_map=at)
