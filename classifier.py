#!/usr/bin/env python3
""" ALL Arduino Nano 33 BLE Sense Classifier

An experiment to explore how low powered microcontrollers, specifically the
Arduino Nano 33 BLE Sense, can be used to detect Acute Lymphoblastic Leukemia.

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

import sys

from abc import ABC, abstractmethod

from modules.AbstractClassifier import AbstractClassifier

from modules.helpers import helpers
from modules.model import model
from modules.server import server


class classifier(AbstractClassifier):
	""" ALL Arduino Nano 33 BLE Sense Classifier

	Represents a HIAS AI Agent that processes data
	using the ALL Arduino Nano BLE Sense Classifier model.
	"""

	def train(self):
		""" Creates & trains the model. """

		self.model.prepare_data()
		self.model.prepare_network()
		self.model.train()
		self.model.evaluate()

	def set_model(self):
		""" Loads the model class """

		self.model = model(self.helpers)

	def load_model(self):
		""" Loads the trained model """

		self.model.load()

	def inference(self):
		""" Classifies test data locally """

		self.load_model()
		self.model.test()

	def server(self):
		""" Loads the API server """

		self.load_model()
		self.server = server(self.helpers, self.model,
							self.model_type)
		self.server.start()

	def inference_http(self):
		""" Classifies test data via HTTP requests """

		self.model.test_http()

	def signal_handler(self, signal, frame):
		self.helpers.logger.info("Disconnecting")
		sys.exit(1)


classifier = classifier()


def main():

	if len(sys.argv) < 2:
		print("You must provide an argument")
		exit()
	elif sys.argv[1] not in classifier.helpers.confs["agent"]["params"]:
		print("Mode not supported! server, train or inference")
		exit()

	mode = sys.argv[1]

	if mode == "train":
		classifier.set_model()
		classifier.train()

	elif mode == "classify":
		classifier.set_model()
		classifier.inference()

	elif mode == "server":
		classifier.set_model()
		classifier.server()

	elif mode == "classify_http":
		classifier.set_model()
		classifier.inference_http()


if __name__ == "__main__":
	main()
