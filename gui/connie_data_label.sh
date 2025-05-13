#!/bin/bash

echo -e "Activating virtualenv\n";
source virtualenv/bin/activate

python3 data_label.py

echo -e "\nDeactivating virtualenv";
deactivate
