import os
import sys
import json
import glob
from collections import Counter
import pdb


# Add current path to python path
sys.path.append(os.getcwd())
import constants


print("-" * 80 + "\n")
print("Measuring of relative frequency of poems by Emily Dickinson began")
print("\n")


input_directory = os.getcwd() + '/' + constants.OUTPUT_FOLDER + \
    'Dickinson_Poems/'

# Get all file names in that directory
input_files = glob.glob(input_directory + '*.*')

for i,element in enumerate(input_files):
    with open(element) as json_file:
        tokens = json.load(json_file)
        file_name = ' '.join(element.split('/')[-1].split('.')[0].split('_')[0:2])
    print ('\n')
    print("Total word count in poem " + str(i+1) + ": "+str(len(tokens)))
    print("The number of occurrences of never in this poem: " + str(Counter(tokens)['never']))
    relative_frequency = Counter(tokens)['never'] / len(tokens)
    print("The relative frequency of never in this poem: " + str(relative_frequency))
