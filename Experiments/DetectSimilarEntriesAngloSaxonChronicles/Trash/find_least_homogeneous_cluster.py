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
from sklearn.metrics import pairwise_distances


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

cluster_ids = set(clusters)
cluster_stdis= []
for element in cluster_ids:
    all_distances= []
    indices = np.where(np.array(clusters)==element)[0]
    entries = [matrix_documents[el] for el in indices]
    entries = np.vstack(entries)
    distances = pairwise_distances(entries)
    for i,element in enumerate(distances):
        dis=distances[i][i+1:].tolist()
        all_distances.extend(dis)
    cluster_stdis.append(np.average(np.array(all_distances)))

print (np.argsort(np.array(cluster_stdis)))
pdb.set_trace()
        
