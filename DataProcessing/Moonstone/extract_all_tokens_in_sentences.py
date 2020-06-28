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
print("Pre-processing of the Moonstone by Wilkie Collins began (takes a few seconds, be patient)")
print("\n")

# Set up the path for the data directory

data_directory = os.getcwd() + '/' + constants.INPUT_FOLDER + \
    'Moonstone/'

output_directory = os.getcwd() + '/' + constants.OUTPUT_FOLDER + \
    'Moonstone/'

nlp = spacy.load("en_core_web_sm")


# Get all file names in that directory
input_files = glob.glob(data_directory + '*.*')

for element in input_files:
    # Read the text
    text = open(element).read()

    # Split for sentences
    spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = len(text)+10000


    tokens_without_stopwords = []
    tokens_with_stopwords = []
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    doc = nlp(text,disable=['parser','ner'])



    sentences_without_stopwords = []
    sentences_with_stopwords = []
    for sentence in doc.sents:

    # Tokenize the text, filter punctuation and transform to lowercase
        
        tokens_without_stopwords = []
        tokens_with_stopwords = []
        
        tokens_without_stopwords.extend([token.lemma_.lower() for token in sentence if (
            (token.pos_ != "PUNCT") and (token.pos_ != "SPACE") and 
            (token.lower_ not in spacy_stopwords)
        )])

        tokens_with_stopwords.extend([token.lemma_.lower()  for token in sentence if (
            (token.pos_ != "PUNCT") and (token.pos_ != "SPACE")
        )])
        sentences_with_stopwords.append(tokens_with_stopwords)
        sentences_without_stopwords.append(tokens_without_stopwords)


    
    # Save results into the output folder

    # Create the output file name

    output_file_with_stopwords = element.split('/')[-1].split('.')[0] + '_all_tokens_in_sentences_with_stopwords.json'
    output_file_without_stopwords = element.split('/')[-1].split('.')[0] + '_all_tokens_in_sentences_without_stopwords.json'

    with open(output_directory + output_file_without_stopwords, 'w') as outfile:
        json.dump(sentences_without_stopwords, outfile)

    with open(output_directory + output_file_with_stopwords, 'w') as outfile:
        json.dump(sentences_with_stopwords, outfile)

print("Pre-processing of the Moonstone by Wilkie Collins finished")
print ("Results are saved to the following folder and files")
print (output_directory)
print (output_file_with_stopwords)
print (output_file_without_stopwords)
