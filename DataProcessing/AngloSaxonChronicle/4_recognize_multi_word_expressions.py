"""Recognizes multi-word expressions and creates a new document collection."""
import os
import json
import pdb
import sys
from gensim.models.phrases import Phraser

# Add current path to python path
sys.path.append(os.getcwd())
from Utils import gensim_utils
import constants

print("-" * 80 + "\n")
print("Recognition of multi-word expressions began")
print("\n")
# Set up the path for the data directory
data_directory = os.getcwd() + '/' + constants.OUTPUT_FOLDER + \
    'Anglo_Saxon_Chronicles/'
input_file_name = "lemmatized_content_words_only_document_collection.json"
output_file_name = "lemmatized_content_words_only_multi_words_document_collection.json"

# Read the document collection bows of lemmatized content words
with open(data_directory + input_file_name) as json_file:
    document_collection = json.load(json_file)

all_sentences = []

# Get all sentences in the corpus

for entry in document_collection:
    for sentence in entry:
        all_sentences.append(sentence)

# Train the phrase model
phrase_model = gensim_utils.build_gensim_phrase_model_from_sentences(
    all_sentences, 5, 3)
phraser_model = Phraser(phrase_model)

# Rebuild the document collection with multiwords
new_document_collection = []
all_terms = []
multi_word_expressions = []

for entry in document_collection:
    entry_sentences = []
    for sentence in entry:
        new_sentence = phraser_model[sentence]
        [multi_word_expressions.append(token) for token in new_sentence if "_"
         in token]
        [all_terms.append(token) for token in new_sentence]
        entry_sentences.append(new_sentence)
    new_document_collection.append(entry_sentences)

phraser_model.save(data_directory + 'phraser_model')

with open(data_directory + output_file_name,
          'w') as file:
    file.write(json.dumps(new_document_collection))

print("The total number of types (lemmas) after recognition of multiwords: " + str(len(set(all_terms))))
print("\n")
print("The total number of recognized multi-words: " + str(len(all_terms)))
print("Recognition of multi-word expressions finished, a new document collection was saved into: " + data_directory + output_file_name)
print("\n")

