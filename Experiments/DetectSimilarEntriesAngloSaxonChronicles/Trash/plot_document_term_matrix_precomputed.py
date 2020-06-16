"""Creates a PCA plot of document term matrix."""


import json
import os
import pdb
import sys
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from plotly import express as px
import plotly
import collections
from sklearn.manifold import TSNE
from random import choices


# Add current path to python path
sys.path.append(os.getcwd())
from Utils import gensim_utils
import constants


# Set the input directory
input_directory = os.getcwd() + '/' + constants.OUTPUT_FOLDER + \
    'Anglo_Saxon_Chronicles/Precomputed_data/'

output_file = 't_sney_document_term_matrix_precomputed.html'
output_file_clusters = 't_sney_document_term_matrix_precomputed_clusters.html'

# Post-process index data here for the visualization (shorten entries)
with open(input_directory + 'index_with_selected_vocab.json') as json_file:
    index_data = json.load(json_file)
new_index_data = []
for element in index_data:
    if len(element.split('<br>')) > 30:
        new_index_data.append('<br>'.join(element.split('<br>')[0:30]))
    else:
        new_index_data.append(element)

# Load the features

with open(input_directory + 'features_with_selected_vocab.json') as json_file:
    features = json.load(json_file)

# Load the document term matrix
matrix_documents = np.loadtxt(input_directory + 'document_term_matrix_with_selected_vocab.txt')


# Load clusters

with open(input_directory + 'labels_ap_clustering_selected_features.json') as file:
    clusters = json.load(file)
# Find the largest clusters: print (mode(clusters))

new_clusters = []


# Add random clusters as well

random_clusters = choices([i for i in range(0,len(set(clusters))) if i not in [13,0]],k=1)

# 8 70
for element in clusters:
    if element == 21:
       new_clusters.append(1)
    elif element == 13:
        new_clusters.append(2)
    elif element == 0:
        new_clusters.append(3)
    elif element in random_clusters:
        new_clusters.append(4)
    elif element == 50:
        new_clusters.append(5)
    elif element == 11:
        new_clusters.append(6)
    elif element == 8:
        new_clusters.append(7)
    elif element == 70:
        new_clusters.append(8)

   

    else:
        new_clusters.append(0)


# Load the dataframe that already has the data

principalDf = pd.read_csv(input_directory+"tsney_selected_vocab.csv")

principalDf['clusters'] = new_clusters

# Plot the results of the principal component analysis
fig = px.scatter(principalDf, x="principal_component_2",
                 y="principal_component_1",
                 hover_name="index",hover_data=["text"])

plotly.offline.plot(fig, filename=input_directory + output_file)


# Plot the results of the principal component analysis
fig = px.scatter(principalDf, x="principal_component_2",
                 y="principal_component_1",
                 hover_name="index",color='clusters',hover_data=["text"])

plotly.offline.plot(fig, filename=input_directory + output_file_clusters)


def show_entry(entry_number):
    """Show the complete entry based on the entry number."""
    entry = '\n'.join(index_data[entry_number].split('<br>'))
    print (entry)


def find_terms_connecting_entries(entry_numbers):
    """Find common terms of entries and print them."""
    entries = []
    common_vocab = []
    for element in entry_numbers:
        entry = '\n'.join(index_data[element].split('<br>'))
        lines = index_data[element].split('<br>')
        vocab = [line for line in lines if len(line.strip().split(' ')) == 1]
        common_vocab.extend(vocab)
        entries.append(entry)
    print ([item for item, count in collections.Counter(common_vocab).items() if count == len(entry_numbers)])
    for text in entries:
        print (text)
        print ('\n')


print ('Use the find_terms_connecting_entries function \
with a list of entries to check common elements')

print ('Use the show_entry function to render a complete text')

pdb.set_trace()
