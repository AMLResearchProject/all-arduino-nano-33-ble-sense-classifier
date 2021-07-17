#!/usr/bin/env python3
""" AI Model Data Augmentation Class.

Provides data augmentation methods.

MIT License

Copyright (c) 2021 Asociación de Investigacion en Inteligencia Artificial
Para la Leucemia Peter Moss

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

from numpy.random import seed

class augmentation():
	""" AI Model Data Augmentation Class

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
		pass

	def equalize_hist(self, data):
		""" Creates a histogram equalized copy. """
		pass

	def reflection(self, data):
		""" Creates a reflected copy. """
		pass

	def gaussian(self, data):
		""" Creates a gaussian blurred copy. """
		pass

	def translate(self, data):
		""" Creates transformed copy. """
		pass

	def rotation(self, data, label, tdata, tlabels):
		""" Creates rotated copies. """
		pass

	def shear(self, data):
		""" Creates a sheared copy. """
		pass