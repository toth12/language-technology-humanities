import os
import spacy
import sys
import json
import glob
import pdb


# Add current path to python path
sys.path.append(os.getcwd())
import constants

sp = spacy.load('en_core_web_sm')
sp.max_length = 10000000


print("-" * 80 + "\n")
print("Pre-processing of the Victorian novels mini corpus began (it takes a few minutes, be patient)")
print("\n")

# Set up the path for the data directory

data_directory = os.getcwd() + '/' + constants.INPUT_FOLDER + \
    'VictorianNovels/'

output_directory = os.getcwd() + '/' + constants.OUTPUT_FOLDER + \
    'VictorianNovels/'


# Get all file names in that directory
input_files = glob.glob(data_directory + '*.*')
outputfiles = []
for element in input_files:
    # Read the text
    text = open(element).read()
    
    # Split for paragraphs first

    paragraphs = text.split('\n')
    paragraphs = list(filter(None, paragraphs))
    spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS

    tokens_without_stopwords = []
    doc = sp.pipe(paragraphs,disable=['parser','ner'])
    

    for paragraph in doc:

# Tokenize the text, filter punctuation and transform to lowercase

        tokens_without_stopwords.extend([token.lemma_.lower() for token in paragraph if (
            (token.pos_ != "PUNCT") and (token.pos_ != "SPACE") and 
            (token.lower_ not in spacy_stopwords)
        )])

        

    
    # Save results into the output folder

    # Create the output file name

    output_file_without_stopwords = element.split('/')[-1].split('.')[0] + '_all_tokens_without_stopwords.json'

    with open(output_directory + output_file_without_stopwords, 'w') as outfile:
        json.dump(tokens_without_stopwords, outfile)
    outputfiles.append(output_file_without_stopwords)

    

print("Pre-processing of the Victorian novels mini corpus finished")
print ("Results are saved to the following folder and files")
print (output_directory)
for el in outputfiles:
    print (el)
