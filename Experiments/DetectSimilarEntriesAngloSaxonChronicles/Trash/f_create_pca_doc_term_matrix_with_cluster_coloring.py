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
from statistics import mode
import re



# Add current path to python path
sys.path.append(os.getcwd())
from Utils import gensim_utils
import constants


# Set the input directory
input_directory = os.getcwd() + '/' + constants.OUTPUT_FOLDER + \
    'Anglo_Saxon_Chronicles/'

# Post-process index data here for the visualization (shorten entries)
with open(input_directory + 'index_with_preselected_verbs.json') as json_file:
    index_data = json.load(json_file)
new_index_data = []
for element in index_data:
    re.findall('\d+|$', '')[0]
    if len(element.split('<br>')) > 30:
        new_index_data.append('<br>'.join(element.split('<br>')[0:30]))
    else:
        new_index_data.append(element)

# Load the document term matrix
matrix_documents = np.loadtxt(input_directory + 'document_term_matrix_with_preselected_verbs.txt')

# Load clusters

with open(input_directory + 'clusters_with_preselected_verbs.json') as file:
    clusters = json.load(file)
# Find the largest clusters: print (mode(clusters))

new_clusters = []
#21 50
for element in clusters:
    if element == 17:
        new_clusters.append(1)
    elif element == 9:
        new_clusters.append(2)
    else:
        new_clusters.append(0)


# Create a principal component analysis
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(matrix_documents)
principalDf = pd.DataFrame(data=principalComponents,
                           columns=['principal_component_1',
                                    'principal_component_2'])
principalDf['text'] = new_index_data
principalDf['index'] = principalDf.index

principalDf['clusters'] = new_clusters
labels = principalDf['index'].tolist()


# Plot the results of the principal component analysis
fig = px.scatter(principalDf, x="principal_component_2",
                 y="principal_component_1",
                 hover_name="index", hover_data=["text"],color='clusters')

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
