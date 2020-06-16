"""Eliminates terms that are not content words.

Returns a list of documents that are lists of sentences that are lists of
content words as lemmas.

"""

import os
import spacy
import sys
import json


# Add current path to python path
sys.path.append(os.getcwd())
import constants
import pdb


print("-" * 80 + "\n")
print("Feature Selection began")
print("Limiting the vocabulary into lemmatized content words began (takes a few seconds, be patient)")
print("\n")
# Set up the path for the data directory
data_directory = os.getcwd() + '/' + constants.OUTPUT_FOLDER + \
    'Anglo_Saxon_Chronicles/'

# Open and read the document collection
with open(data_directory + 'document_collection.json') as json_file:
    document_collection = json.load(json_file)

# Load project specific stopwords,blacklisted pos and tags
with open(data_directory +
          'Precomputed_data/project_specific_stop_words_pos_tags.json') as json_file:
    project_specific_stop_words = json.load(json_file)

stopwords = project_specific_stop_words['stopwords']
blacklisted_pos = project_specific_stop_words['blacklisted_pos']
blacklisted_tags = project_specific_stop_words['blacklisted_tags']

# Preprocess the document collection with Spacy
sp = spacy.load('en_core_web_sm')

# Create an empty list that will hold the results
linguistically_preprocessed_document_collection = []

collection = sp.pipe(document_collection)
lemmatized_content_words_only_document_collection = []
all_content_words = []

for i, entry in enumerate(collection):
    # Iterate through all tokens of an entry
    entry_sentences = []
    for sententence in entry.sents:
        new_sentence = []
        for token in sententence:
            if ((token.pos_ not in blacklisted_pos) and
               (token.tag_ not in blacklisted_tags) and
               (token.lemma_.lower() not in stopwords)):
                if '-' in token.lemma_:
                    all_content_words.append(token.text)
                    new_sentence.append(token.text)
                else:
                    all_content_words.append(token.lemma_)
                    new_sentence.append(token.lemma_)




        entry_sentences.append(new_sentence)
    lemmatized_content_words_only_document_collection.append(entry_sentences)


with open(data_directory +
          'lemmatized_content_words_only_document_collection.json',
          'w') as file:
    file.write(json.dumps(lemmatized_content_words_only_document_collection))


print("The total number of tokens (replaced for lemmas) after selection of content words: " + str(len(all_content_words)))
print("The total number of types (lemmas) after selection of content words: " + str(len(set(all_content_words))))
print("\n")
print("Limiting the vocabulary into lemmatized content words finished; results saved to: " + data_directory +'lemmatized_content_words_only_document_collection.json')

print("Feature Selection finished")
print ("Result is written to the following file and folder")
print ('lemmatized_content_words_only_document_collection.json')
print (data_directory)

