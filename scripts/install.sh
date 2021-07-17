#!/bin/bash

FMSG="Acute Lymphoblastic Leukemia Arduino Nano 33 BLE Sense Classifier trainer installation terminated!"

echo "This script will install Acute Lymphoblastic Leukemia Arduino Nano 33 BLE Sense Classifier."

read -p "Proceed (y/n)? " proceed
if [ "$proceed" = "Y" -o "$proceed" = "y" ]; then
# DEVELOPER TO ADD INSTALLATION COMMANDS FOR ALL REQUIRED LIBRARIES (apt/pip etc)
else
	echo $FMSG;
	exit 1
fi