"""Trains a TF-IDF model of all entries and returns a document term matrix.

Multiword expressions are recognized;project specific stopwords eliminated
all terms the df of which is below 5 removed; documents that are outliers
(too few or too many resulting features after training) removed; importance
of features are measured with a RandomForestClassifier and unimportant features
removed from the document term matrix.
"""

import os
import re
import spacy
import json
import pdb
import sys
import pandas as pd
from gensim.models.phrases import Phraser
from gensim.models import TfidfModel
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing


# Add current path to python path
sys.path.append(os.getcwd())
from Utils import gensim_utils
import constants

# Parse files in the input folder
input_directory = os.getcwd() + '/' + constants.INPUT_FOLDER + \
    'Anglo_Saxon_Chronicles/'

# Set up the output directory
output_directory = os.getcwd() + '/' + constants.OUTPUT_FOLDER + \
    'Anglo_Saxon_Chronicles/'

# Read the input file
complete_text = open(input_directory + 'anglo_saxon_chronicle.txt').read()
# Get each year (entries) from the chronicle
entries = complete_text.split('\n\n')
# Eliminate line breaks from each entry
entries_without_line_break = [' '.join(element.split('\n'))
                              for element in entries]

# Load the spacy model
sp = spacy.load('en_core_web_sm')

# Load project specific stopwords,blacklisted pos and tags
with open(output_directory +
          'project_specific_stop_words_pos_tags.json') as json_file:
    project_specific_stop_words = json.load(json_file)

stopwords = project_specific_stop_words['stopwords']
blacklisted_pos = project_specific_stop_words['blacklisted_pos']
blacklisted_tags = project_specific_stop_words['blacklisted_tags']

# Load the phraser model trained in the previous step
phraser_model = Phraser.load(output_directory + 'phraser_model')

# Create a list that will hold the bag of word representation of each entry
bows = []

# Create a bag of word representation of each entry
for i, entry in enumerate(entries_without_line_break):
    # POS tag each entry
    entry_text = sp(entry)
    bow = []
    for sentence in entry_text.sents:
        sentence_element = []
        # Remove blacklisted elements
        for token in sentence:
            if ((token.pos_ not in blacklisted_pos) and
               (token.tag_ not in blacklisted_tags) and (token.lemma_.lower()
               not in stopwords)):
                if '-' in token.lemma_:
                    sentence_element.append(token.text)
                else:
                    sentence_element.append(token.lemma_)
        # Apply the phraser model to get multi word expressions,update bow
        bow.extend(phraser_model[sentence_element])
    bows.append(bow)

preselected_features = pd.read_csv(output_directory +'historically_specific_vocab.csv')['0'].to_list()


# Initialize a gensim dictionary with bows
gensim_dic = gensim_utils.initialize_gensim_dictionary([preselected_features ])

# Create a TF-IDF based bag of word representation of each entry


# Build a gensim corpus
gensim_corpus = [gensim_dic.doc2bow(text) for text in bows]

# Train a tf-idf model and reuse the gensim dic
id2word={key: value for (key, value) in enumerate(preselected_features)}
model = TfidfModel(gensim_corpus, id2word=id2word,normalize=True)

# Build a tf-idf corpus
corpus_tfidf = model[gensim_corpus]

# Get the matrix representation (matrix_documents) of the TF-IDF corpus
n_items = len(gensim_dic)
ds = []
for doc in corpus_tfidf:
    d = [0] * n_items
    for index, value in doc:
        d[index] = value
    ds.append(d)
matrix_documents = np.array(ds)

# Remove entries that have many null values or too many values
entries_for_remove = []
for element in enumerate(matrix_documents):
    if ((np.nonzero(element[1])[0].shape[0] < 5) or
       (np.nonzero(element[1])[0].shape[0] > 100)):
        entries_for_remove.append(element[0])
matrix_documents = np.delete(matrix_documents, entries_for_remove, 0)


# Create a list from the features (features_list) so that they can be reused
features_list = [element[1] for element in gensim_dic.iteritems()]



np.savetxt(output_directory + 'document_term_matrix_with_preselected_vocab.txt', matrix_documents)

# Create a new entry index by replacing line break with <br> tag
# Add line breaks to each entry
entries = complete_text.split('\n\n')
entries_without_line_break = ['<br> '.join(element.split('\n'))
                              for element in entries]
# Remove too long spaces
entries_without_line_break = [re.sub(' +', ' ', element) for element
                              in entries_without_line_break]

# Create an empty index that will hold them
index = []

pdb.set_trace()

# Remove those entries that were removed from document term matrix
for i in sorted(entries_for_remove, reverse=True):
    del entries_without_line_break[i]

# Add the terms in bow of each entry
for i, entry in enumerate(entries_without_line_break):
    # Find the non zero element in each row of the document term matrix
    vocab = np.nonzero(matrix_documents[i])[0].tolist()
    # Add them to the index data
    for element in vocab:
        entry = entry + '<br>' + features_list[element]
    index.append(entry)

# Save the updated index data
with open(output_directory + 'index_with_preselected_vocab.json', 'w') as file:
    file.write(json.dumps(index))

pdb.set_trace()
