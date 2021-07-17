#!/usr/bin/env python
""" Abstract class representing a HIAS AI Agent.

Represents a HIAS AI Agent. HIAS AI Agents process data using local AI
models and communicate with HIAS IoT Agents using the MQTT protocol.

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

import psutil
import requests
import threading

from abc import ABC, abstractmethod

from modules.helpers import helpers
from modules.mqtt import mqtt
from modules.model import model

from threading import Thread


class AbstractAgent(ABC):
	""" Abstract class representing a HIAS AI Agent.

	This object represents a HIAS AI Agent. HIAS AI Agents
	process data using local AI models and communicate
	with HIAS IoT Agents using the MQTT protocol.

	Attributes
	----------
	NA

	Methods
	-------
	mqtt_conn()
		Creates a MQTT connection with the HIAS iotJumpWay
		private MQTT broker.
	"""

	def __init__(self):
		"Initializes the abstract_agent object."
		super().__init__()

		self.mqtt = None
		self.helpers = helpers("Agent")
		self.confs = self.helpers.confs
		self.model_type = None

		self.helpers.logger.info("Agent initialization complete.")

	def mqtt_conn(self, credentials):
		""" Starts the HIAS iotJumpWay MQTT broker connection. """

		self.mqtt = mqtt(self.helpers, "Agent", credentials)
		self.mqtt.configure()
		self.mqtt.start()

		#self.mqtt.subscribe()
		self.mqtt.commands_callback = self.mqtt_commands

		self.agent_threading()

		self.helpers.logger.info(
			"HIAS iotJumpWay MQTT Broker connection created and subscriptions created.")

	def mqtt_start(self):
		""" Starts the HIAS iotJumpWay MQTT broker connection. """

		self.mqtt_conn({
			"host": self.credentials["iotJumpWay"]["host"],
			"port": self.credentials["iotJumpWay"]["port"],
			"location": self.credentials["iotJumpWay"]["location"],
			"zone": self.credentials["iotJumpWay"]["zone"],
			"entity": self.credentials["iotJumpWay"]["entity"],
			"name": self.credentials["iotJumpWay"]["name"],
			"un": self.credentials["iotJumpWay"]["un"],
			"up": self.credentials["iotJumpWay"]["up"]
		})

	def mqtt_commands(self):
		""" Called in the event of a command message from the HIAS iotJumpWay MQTT broker. """

	def life(self):
		""" Publishes entity statistics to HIAS. """

		cpu = psutil.cpu_percent()
		mem = psutil.virtual_memory()[2]
		hdd = psutil.disk_usage('/').percent

		if self.model_type == "IR":
			tmp = psutil.sensors_temperatures()['cpu_thermal'][0].current
		else:
			tmp = psutil.sensors_temperatures()['coretemp'][0].current
		r = requests.get('http://ipinfo.io/json?token=' +
					self.helpers.credentials["iotJumpWay"]["ipinfo"])
		data = r.json()
		if data["status"] != 403:
			location = data["loc"].split(',')
		else:
			location = [0, 0]

		self.mqtt.publish("Life", {
			"CPU": float(cpu),
			"Memory": float(mem),
			"Diskspace": float(hdd),
			"Temperature": float(tmp),
			"Latitude": float(location[0]),
			"Longitude": float(location[1])
		})

		self.helpers.logger.info("Agent life statistics published.")
		threading.Timer(300.0, self.life).start()

	def agent_threading(self):
		""" Creates required module threads. """

		# Life thread
		threading.Timer(10.0, self.life).start()

	@abstractmethod
	def set_model(self):
		""" Creates & trains the model. """
		pass

	@abstractmethod
	def train(self):
		""" Creates & trains the model. """
		pass

	@abstractmethod
	def load_model(self):
		""" Loads model and classifies test data locally """
		pass

	@abstractmethod
	def inference(self):
		""" Loads model and classifies test data """
		pass