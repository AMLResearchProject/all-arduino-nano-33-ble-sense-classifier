#!/usr/bin/env python3
""" Server/API class.

Class for the classifier server/API.

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

import cv2
import json
import jsonpickle
import os
import requests
import time

import numpy as np

from io import BytesIO
from PIL import Image
from flask import Flask, request, Response

from modules.AbstractServer import AbstractServer

class server(AbstractServer):
	""" Server/API class.

	Class for the classifier server/API.
	"""

	def predict(self, req):
		""" Classifies an image sent via HTTP. """

		if len(req.files) != 0:
			img = Image.open(req.files['file'].stream)
		else:
			img = Image.open(BytesIO(req.data))

		return self.model.predict(self.model.http_reshape(img))

	def start(self):
		""" Starts the server. """

		app = Flask("AllANBS")

		@app.route('/Inference', methods=['POST'])
		def Inference():
			""" Responds to HTTP POST requests. """

			prediction = self.predict(request)

			if prediction == 1:
				message = "Acute Lymphoblastic Leukemia detected!"
				diagnosis = "Positive"
			elif prediction == 0:
				message = "Acute Lymphoblastic Leukemia not detected!"
				diagnosis = "Negative"

			resp = jsonpickle.encode({
				'Response': 'OK',
				'Message': message,
				'Diagnosis': diagnosis
			})

			return Response(response=resp, status=200, mimetype="application/json")

		app.run(host=self.helpers.confs["agent"]["ip"],
				port=self.helpers.confs["agent"]["port"])