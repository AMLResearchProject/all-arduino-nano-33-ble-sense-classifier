#!/usr/bin/env python3
""" Abstract class representing a HIAS AI Model.

Represents an AI Model. HIAS AI Models are used by AI Agents to process
incoming data. Based on HIAS AI Models for future compatibility with
the HIAS Network.

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
- Adam Milton-Barker - First version - 2021-5-1

"""

import os
import random

from numpy.random import seed

from abc import ABC, abstractmethod

from modules.data import data

class AbstractModel(ABC):
	""" Abstract class representing an AI Model.

	Represents an AI Model. HIAS AI Models are used by AI Agents
	to process incoming data. Based on HIAS AI Models for future
	compatibility with the HIAS Network.
	"""

	def __init__(self, helpers):
		""" Initializes the AbstractModel object. """
		super().__init__()

		self.helpers = helpers
		self.confs = self.helpers.confs

		os.environ["KMP_BLOCKTIME"] = "1"
		os.environ["KMP_SETTINGS"] = "1"
		os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
		os.environ["OMP_NUM_THREADS"] = str(
			self.confs["agent"]["cores"])

		self.data = data(self.helpers)

		self.testing_dir = self.confs["data"]["test"]
		self.valid = self.confs["data"]["valid_types"]
		self.seed = self.confs["data"]["seed"]

		random.seed(self.seed)
		seed(self.seed)

		self.weights_file = self.confs["model"]["weights"]
		self.json_model = self.confs["model"]["model"]
		self.tflite_model = self.confs["model"]["tfmodel"]
		self.c_array_model = self.confs["model"]["model_c_array"]

		self.helpers.logger.info("Model class initialization complete.")

	@abstractmethod
	def prepare_data(self):
		""" Prepares the model data """
		pass

	@abstractmethod
	def prepare_network(self):
		""" Builds the network """
		pass

	@abstractmethod
	def train(self):
		""" Trains the model """
		pass

	@abstractmethod
	def save_model_as_json(self):
		""" Saves the model as JSON """
		pass

	@abstractmethod
	def save_weights(self):
		""" Saves the model weights """
		pass

	@abstractmethod
	def evaluate(self):
		""" Evaluates the model """
		pass

	@abstractmethod
	def visualize_metrics(self):
		""" Visualize the metrics. """
		pass

	@abstractmethod
	def confusion_matrix(self):
		""" Prints/displays the confusion matrix. """
		pass

	@abstractmethod
	def figures_of_merit(self):
		""" Calculates/prints the figures of merit. """
		pass

	@abstractmethod
	def predictions(self):
		""" Makes predictions on the train & test sets. """
		pass

	@abstractmethod
	def predict(self, img):
		""" Gets a prediction for an image. """
		pass

	@abstractmethod
	def reshape(self, img):
		""" Reshapes an image. """
		pass

	@abstractmethod
	def test(self):
		"""Local test mode

		Loops through the test directory and classifies the images.
		"""
		pass