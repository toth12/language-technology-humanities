"""Creates a T-SNEY plot of the document term matrix.

Saves the plot into Data/Outputs/t_sney_document_term_matrix.html
Contains utility functions to access information behind the plot.

"""


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
import constants


print("-" * 80 + "\n")
print("Rendering the Feature Space representing the Anglo-Saxon Chronicle Through Projection to a Lower-Dimensional Space began")
print("\n")



# Set the input directory
data_directory = os.getcwd() + '/' + constants.OUTPUT_FOLDER + \
    'Anglo_Saxon_Chronicles/'

# Post-process index data here for the visualization (shorten entries)
input_file = 'post_processed_document_collection.json'
output_file = 't_sney_document_term_matrix_new.html'
with open(data_directory + input_file) as json_file:
    index_data = json.load(json_file)
new_index_data = []
for element in index_data:
    if len(element.split('<br>')) > 30:
        new_index_data.append('<br>'.join(element.split('<br>')[0:30]))
    else:
        new_index_data.append(element)

# Load the features
with open(data_directory +
          'features_list_reshaped_TF_IDF_training.json') as json_file:
    features = json.load(json_file)

# Load the document term matrix
documents_terms_matrix = np.loadtxt(data_directory +
                                    'document_term_matrix_TF_IDF.txt')

# Load clusters
with open(data_directory + 'clusters.json') as file:
    clusters = json.load(file)

# Add cluster membership to five selected clusters

new_clusters = []
random_clusters = (choices([i for i in
                   range(0, len(set(clusters))) if i not in [13, 0]], k=1))

# 8 70
for element in clusters:
    if element == 6:
        new_clusters.append(1)
    elif element == 4:
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

# Create a principal component analysis
pca = PCA(n_components=50)
principalComponents = (TSNE().fit_transform(
                       (pca.fit_transform(documents_terms_matrix)
                        )))
principalDf = pd.DataFrame(data=principalComponents,
                           columns=['principal_component_1',
                                    'principal_component_2'])

principalDf['clusters'] = new_clusters
principalDf['text'] = new_index_data
principalDf['index'] = principalDf.index


labels = principalDf['index'].tolist()

# Plot the results of the principal component analysis without clusters
fig = px.scatter(principalDf, x="principal_component_2",
                 y="principal_component_1",
                 hover_name="index", hover_data=["text"])

plotly.offline.plot(fig, filename=data_directory + output_file)

# Plot the results of the principal component analysis with clusters
fig = px.scatter(principalDf, x="principal_component_2",
                 y="principal_component_1",
                 hover_name="index", color='clusters',hover_data=["text"])

plotly.offline.plot(fig, filename=data_directory + 't_sney_document_term_matrix_with_clusters_new.html')


def show_entry(entry_number):
    """Show the complete entry based on the entry number."""
    entry = '\n'.join(index_data[entry_number].split('<br>'))
    print(entry)


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
    print([item for item, count in collections.Counter(common_vocab).items()
          if count == len(entry_numbers)])
    for text in entries:
        print(text)
        print('\n')


print ("The output of the visualizations saved to the following folder:\n")
print (data_directory)
print ("\n")
print ("The output of the visualizations saved to the following two files in the directory above:\n")
print ("1. t_sney_document_term_matrix_with_clusters_new.html\n2. t_sney_document_term_matrix_new.html\n")
print ("Examine the two visualizations; by mouse hoovering over a point, you will see the corresponding entry; it is sometimes abbreviated.")
print ("\n")
print("Use the show_entry function to render a complete entry; for instance to render entry number two, type 'show_entry(2)")
print ("\n")
print ("If you want to continue with the process, type 'c'")



pdb.set_trace()
