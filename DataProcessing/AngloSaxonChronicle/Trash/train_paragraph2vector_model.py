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
from gensim.models.doc2vec import LabeledSentence
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


def chunks(l, n):
    
    l = [element for element in l if len(element) > 0]
    # For item i in a range that is a length of l,
    
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i+n]

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
index = []

# Create an index
for i, entry in enumerate(entries_without_line_break):
    # POS tag each entry
    entry_text = sp(entry)
    sentences = []
    for sentence in entry_text.sents:
        sentences.append(sentence.text.strip())
    chunk = [element for element in chunks(sentences,1)]
    [index.append('<br>'.join(doc)) for doc in chunk]
# Create a bag of word representation of each entry

for i, entry in enumerate(index):
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
        bow.append(phraser_model[sentence_element])
    # Chunk the document into sequences of three sentences
    chunk = [element for element in chunks(bow,1)]

    # If last sequence is only 1, add it to the previous chunk, and delete it

    '''if len(chunk) > 1:
        if len(chunk[len(chunk)-1]) == 1:
            chunk[len(chunk)-2].extend(chunk[len(chunk)-1])
            del chunk[len(chunk)-1] '''
    # Create Labeled Sentences from each chunk
    for element in chunk:
        chunk_bow = [ ]
        [chunk_bow.extend(doc) for doc in element]
        label_sentence = TaggedDocument(words=chunk_bow,tags=[i])
        bows.append(label_sentence)
model_doc2vec = Doc2Vec(bows,min_count=2,epochs=100)
# Create a document term matrix

# Normalize the document vectors
#model_doc2vec.docvecs.init_sims(replace=True)
all_docs = []
for f,document in enumerate(model_doc2vec.docvecs.vectors_docs):

        #add it to processed documents
        all_docs.append(model_doc2vec.docvecs[f])

#create one array from the processed_documents
document_term_matrix = np.vstack(all_docs)
pdb.set_trace()
np.savetxt(output_directory + 'document_term_matrix_chunks.txt', document_term_matrix)


# Save the updated index data
with open(output_directory + 'index_chunks.json', 'w') as file:
    file.write(json.dumps(index))