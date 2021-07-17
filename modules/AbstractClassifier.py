#!/usr/bin/env python3
""" Abstract class representing an AI Classifier.

Represents an AI Classifier. AI Classifiers process data using AI
models. Based on HIAS AI Agents for future compatibility with
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
- Adam Milton-Barker

"""

from abc import ABC, abstractmethod

from modules.helpers import helpers
from modules.model import model


class AbstractClassifier(ABC):
	""" Abstract class representing an AI Classifier.

	Represents an AI Classifier. AI Classifiers process data using AI
	models. Based on HIAS AI Agents for future compatibility with
	the HIAS Network.
	"""

	def __init__(self):
		""" Initializes the AbstractClassifier object. """
		super().__init__()

		self.helpers = helpers("Classifier")
		self.confs = self.helpers.confs
		self.model_type = None

		self.helpers.logger.info("Classifier initialization complete.")

	@abstractmethod
	def set_model(self):
		""" Loads the model class """
		pass

	@abstractmethod
	def train(self):
		""" Creates & trains the model. """
		pass

	@abstractmethod
	def load_model(self):
		""" Loads the AI model """
		pass

	@abstractmethod
	def inference(self):
		""" Loads model and classifies test data """
		pass

	@abstractmethod
	def server(self):
		""" Loads the API server """
		pass

	@abstractmethod
	def inference_http(self):
		""" Classifies test data via HTTP requests """
		pass