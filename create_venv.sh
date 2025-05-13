#!/bin/bash

echo "Creating virtualenv";
python3 -m venv virtualenv

echo "Activating virtualenv";
source virtualenv/bin/activate

echo "Installing requirements";
pip3 install --upgrade pip
pip3 install -r requirements.txt

echo "Deactivating virtualenv";
deactivate
