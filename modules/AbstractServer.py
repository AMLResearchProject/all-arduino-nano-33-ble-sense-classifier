#!/usr/bin/env python3
""" Server/API abstract class.

Abstract class for the classifier server/API.

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

class AbstractServer(ABC):
	""" Server/API abstract class.

	Abstract class for the classifier server/API.
	"""

	def __init__(self, helpers, model, model_type):
		"Initializes the AbstractServer object."
		super().__init__()

		self.helpers = helpers
		self.confs = self.helpers.confs

		self.model = model
		self.model_type = model_type

		self.helpers.logger.info("Server initialization complete.")


	@abstractmethod
	def predict(self, req):
		""" Classifies an image sent via HTTP. """
		pass

	@abstractmethod
	def start(self, img_path):
		""" Sends image to the inference API endpoint. """
		pass
