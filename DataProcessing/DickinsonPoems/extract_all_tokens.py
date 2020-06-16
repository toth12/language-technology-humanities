import os
import spacy
import sys
import json
import glob


# Add current path to python path
sys.path.append(os.getcwd())
import constants

sp = spacy.load('en_core_web_sm')
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS


print("-" * 80 + "\n")
print("Pre-processing of two poems by Emily Dickinson began")
print("\n")

# Set up the path for the data directory

data_directory = os.getcwd() + '/' + constants.INPUT_FOLDER + \
    'Dickinson_Poems/'

output_directory = os.getcwd() + '/' + constants.OUTPUT_FOLDER + \
    'Dickinson_Poems/'


# Get all file names in that directory
input_files = glob.glob(data_directory + '*.*')
outfiles =[]
for element in input_files:
    # Read the text
    poem_text = open(element).read()

    # Tokenize the text, filter punctuation and transform to lowercase

    tokens_without_stop_words = [token.lower_ for token in sp(poem_text) if (
        (token.pos_ != "PUNCT") and (token.pos_ != "SPACE")
    )]

    # Save results into the output folder

    # Create the output file name

    output_file = element.split('/')[-1].split('.')[0] + '_all_tokens.json'
    outfiles.append(output_file)
    with open(output_directory + output_file, 'w') as outfile:
        json.dump(tokens_without_stop_words, outfile)

print("Pre-processing of two poems by Emily Dickinson finished")
print("Resuts saved into the following folder and files")
print (output_directory)
for el in outfiles:
    print (el)
