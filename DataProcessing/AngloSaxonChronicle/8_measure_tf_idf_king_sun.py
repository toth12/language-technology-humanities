"""Trains a non-normalized TF-IDF model of the Anglo-Saxon Chronicle.

It prints the normalized document frequency, inverse document frequency and the tf-idf of certain terms

"""

import os
import json
import sys
import pandas as pd
from gensim.models import TfidfModel
import numpy as np
import pdb
import spacy
from gensim.models.phrases import Phraser
import pdb

# Add current path to python path
sys.path.append(os.getcwd())
from Utils import gensim_utils
import constants

# Set up the path for the data directory
data_directory = os.getcwd() + '/' + constants.OUTPUT_FOLDER + \
    'Anglo_Saxon_Chronicles/'
precomputed_directory = data_directory + "Precomputed_data/"
input_file_name = 'lemmatized_content_words_only_multi_words_document_\
collection.json'

# Load the selected features
preselected_features = (pd.read_csv(data_directory +
                        'Precomputed_data/historically_specific_vocab.csv')['0'].to_list())

# Load the phraser model trained in the previous step
phraser_model = Phraser.load(data_directory + 'phraser_model')

# Flatten the preselected features

flattened_features = []

for element in preselected_features:
    if '_' in element:
        new_element = element.split('_')
        flattened_features.extend(new_element)
    else:
        flattened_features.append(element)


# Load the preprocessed data
with open(data_directory +
          input_file_name) as json_file:
    preprocessed_document_collection = json.load(json_file)

'''Transform the preprocessed data (each entry is a list of sentences
that are list of lemmas) to bag of words (each entry is a list of lemmas)
'''
# Open and read the document collection
with open(data_directory + 'document_collection.json') as json_file:
    document_collection = json.load(json_file)

# Preprocess the document collection with Spacy
sp = spacy.load('en_core_web_sm')
collection = sp.pipe(document_collection)

bows = []


# Create a bag of word representation of each entry
for i, entry in enumerate(collection):
    bow = []
    for sentence in entry.sents:
        sentence_element = []
        # Remove blacklisted elements
        for token in sentence:
                if '-' in token.lemma_:
                    if token.text in flattened_features:
                        sentence_element.append(token.text)

                else:
                    if token.lemma_ in flattened_features:
                        sentence_element.append(token.lemma_)
        # Apply the phraser model to get multi word expressions,update bow
        bow.extend(phraser_model[sentence_element])
    bows.append(bow)

# Load the entries to be removed (precomputed earlier)
with open(precomputed_directory + "removed_entries.json") as json_file:
    entries_for_remove = json.load(json_file)

# Remove them from the bow representation
for i in sorted(entries_for_remove, reverse=True):
    del bows[i]
    del document_collection[i]


# Initialize a gensim dictionary with bows
gensim_dic = gensim_utils.initialize_gensim_dictionary([preselected_features])

# Create a TF-IDF based bag of word representation of each entry


# Build a gensim corpus
gensim_corpus = [gensim_dic.doc2bow(text) for text in bows]


# Train a tf-idf model and reuse the gensim dic
#not neededid2word = {key: value for (key, value) in enumerate(preselected_features)}

model = TfidfModel(gensim_corpus,id2word=gensim_dic.id2token,normalize=False)

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


with open(data_directory + 'index_removed_entries.json', 'w') as file:
    file.write(json.dumps(entries_for_remove))


# Create a list from the features (features_list) so that they can be reused
# The order of the loaded features was changed during the training process

features_list = [element[1] for element in gensim_dic.iteritems()]


#Get the indices of certain terms

index_monk = features_list.index('monk')
index_king  = features_list.index('king')
index_sun  = features_list.index('sun')


#calculate document frequency king

#get the column corresponding to king
king_column = matrix_documents[:,index_king ]

#get the number of documents where king occurs
number_of_doc_occurrences_king = len(np.nonzero(king_column)[0])

#divide it by the total number of documents to get the document frequency

df_king = np.divide(number_of_doc_occurrences_king,len(king_column))

#calculate the inverse df of king

idf_king=np.log2(np.divide(len(king_column),number_of_doc_occurrences_king))

tf_idf_king_93_row = idf_king*2
print ("This is the normalized document frequency of king: "+str(df_king))
print ("This is the inverse document frequency of king: "+str(idf_king))
print ("This is the tf-idf of king in the following entry of the Anglo-Saxon Chronicle: "+str(tf_idf_king_93_row))
print("\n")

print(document_collection[93])



#from gensim.models import tfidfmodel


#calculate document frequency sun

#get the column corresponding to sun
sun_column = matrix_documents[:,index_sun ]

#get the number of documents where sun occurs
number_of_doc_occurrences_sun = len(np.nonzero(sun_column)[0])

#divide it by the total number of documents to get the document frequency

df_sun= np.divide(number_of_doc_occurrences_sun,len(sun_column))

# calculate the inverse df of sun

idf_sun=np.log2(np.divide(len(sun_column),number_of_doc_occurrences_sun))
print ("\n")
print ("This is the normalized document frequency of sun: "+str(df_sun))
print ("This is the inverse document frequency of sun: "+str(idf_sun))

pdb.set_trace()

