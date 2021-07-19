""" Helpers file.

Configuration and logging functions.

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

import logging
import logging.handlers as handlers
import json
import os
import sys
import time

from datetime import datetime


class helpers():
	""" Helper Class

	Configuration and logging functions.
	"""

	def __init__(self, ltype, log=True):
		""" Initializes the Helpers Class. """

		# Loads system configs
		self.confs = {}
		self.loadConfs()

		# Sets system logging
		self.logger = logging.getLogger(ltype)
		self.logger.setLevel(logging.INFO)

		formatter = logging.Formatter(
			'%(asctime)s - %(name)s - %(levelname)s - %(message)s')

		allLogHandler = handlers.TimedRotatingFileHandler(
			os.path.dirname(os.path.abspath(__file__)) + '/../logs/all.log', when='H', interval=1, backupCount=0)
		allLogHandler.setLevel(logging.INFO)
		allLogHandler.setFormatter(formatter)

		errorLogHandler = handlers.TimedRotatingFileHandler(
			os.path.dirname(os.path.abspath(__file__)) + '/../logs/error.log', when='H', interval=1, backupCount=0)
		errorLogHandler.setLevel(logging.ERROR)
		errorLogHandler.setFormatter(formatter)

		warningLogHandler = handlers.TimedRotatingFileHandler(
			os.path.dirname(os.path.abspath(__file__)) + '/../logs/warning.log', when='H', interval=1, backupCount=0)
		warningLogHandler.setLevel(logging.WARNING)
		warningLogHandler.setFormatter(formatter)

		consoleHandler = logging.StreamHandler(sys.stdout)
		consoleHandler.setFormatter(formatter)

		self.logger.addHandler(allLogHandler)
		self.logger.addHandler(errorLogHandler)
		self.logger.addHandler(warningLogHandler)
		self.logger.addHandler(consoleHandler)

		if log is True:
			self.logger.info("Helpers class initialization complete.")

	def loadConfs(self):
		""" Load the configuration. """

		with open(os.path.dirname(os.path.abspath(__file__)) + '/../configuration/config.json') as confs:
			self.confs = json.loads(confs.read())
