"""Transforms the Anglo-Saxon Chronicle into document collection."""

import os
from os import listdir
import sys
import json

print("-" * 80 + "\n")
print("The transformation of the Anglo-Saxon Chronicle began.\n")
# Add current path to python path
sys.path.append(os.getcwd())
import constants

# Parse files in the input folder
input_directory = os.getcwd() + '/' + constants.INPUT_FOLDER + \
    'Anglo_Saxon_Chronicles/'
files = listdir(input_directory)

# Set up the output directory
output_directory = os.getcwd() + '/' + constants.OUTPUT_FOLDER + \
    'Anglo_Saxon_Chronicles/'

# Open, read input data and eliminate line breaks
for file in files:
    complete_text = open(input_directory + file).read()
    entries = complete_text.split('\n\n')


# Save the document collection
with open(output_directory + 'document_collection.json', 'w') as file:
    file.write(json.dumps(entries))


print("Results saved into:" + output_directory + 'document_collection.json\n')

print("The number of documents in the docoument collection: " + str(len(entries)) + '\n')

print("The transformation of the Anglo-Saxon Chronicle finished.\n")
print("-" * 80 + "\n")
