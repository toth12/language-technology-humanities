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
import re


# Add current path to python path
sys.path.append(os.getcwd())
from Utils import gensim_utils
import constants


# Set the input directory
input_directory = os.getcwd() + '/' + constants.OUTPUT_FOLDER + \
    'Anglo_Saxon_Chronicles/'

# Post-process index data here for the visualization (shorten entries)
with open(input_directory + 'index_with_selected_vocab.json') as json_file:
    index_data = json.load(json_file)
new_index_data = []

time_index = []
for element in index_data:
    time = re.findall('\d{2,4}', element)
    if len(time) >0:
        time_index.append(time[0])
    else:
        time_index.append(0)

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
size = []
#21 50
for element in clusters:
    size.append(1)
    if element == 36:
        new_clusters.append(1)
    else:
        new_clusters.append(0)
    '''elif element == 162:
        new_clusters.append(2)
    elif element == 69:
        new_clusters.append(3)
    elif element == 74:
        new_clusters.append(4)'''
    



# Load the dataframe that already has the data

principalDf = pd.read_csv("tsney_selected_vocab.csv")

principalDf['clusters'] = new_clusters
principalDf['time'] = time_index
principalDf['size'] = size
pdb.set_trace()

# Plot the results of the principal component analysis
fig = px.scatter_3d(principalDf, x="principal_component_2",
                 y="principal_component_1", z='time',
                 hover_name="index",color='clusters',hover_data=["text"], width=3000, height=1500, size='size')

plotly.offline.plot(fig, filename=input_directory + 'pca.html')


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
