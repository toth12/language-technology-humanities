"""Preprocess the document collection: tokenize, lemmatize, pos tag.

Returns the processed collection (document_collection_linguistically
preprocessed in a python list; each element is a list of
spacy object.
"""

import os
import spacy
import sys
import json


# Add current path to python path
sys.path.append(os.getcwd())
import constants


print("-" * 80 + "\n")
print("Linguistic processing of the Anglo-Saxon Chronicle began (takes a few seconds, be patient)")
print("\n")
# Set up the path for the data directory
data_directory = os.getcwd() + '/' + constants.OUTPUT_FOLDER + \
    'Anglo_Saxon_Chronicles/'

# Open and read the document collection
with open(data_directory + 'document_collection.json') as json_file:
    document_collection = json.load(json_file)

# Preprocess the document collection with Spacy
sp = spacy.load('en_core_web_sm')

# Create an empty list that will hold the results
linguistically_preprocessed_document_collection = []

collection = sp.pipe(document_collection)
complete_vocab_token = []
complete_vocab_lemma = []
for i, entry in enumerate(collection):
    # Iterate through all tokens of an entry
    for token in entry:
        if ((token.pos_ != "PUNCT") and (token.pos_ != "SPACE")):
            complete_vocab_token.append(token.text)
            if '-' in token.lemma_:
                complete_vocab_lemma.append(token.text)
            else:
                complete_vocab_lemma.append(token.lemma_)

print("Total token count (without punctuation): " +
      str(len(complete_vocab_token)))
print("Total type count (without punctuation): " +
      str(len(set(complete_vocab_token))))
print("\n")
print("Total type (lemma) count (without punctuation): " +
      str(len(set(complete_vocab_lemma)))
      )
