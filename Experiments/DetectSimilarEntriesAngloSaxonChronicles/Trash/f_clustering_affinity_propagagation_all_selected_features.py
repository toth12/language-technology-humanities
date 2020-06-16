"""Creates a affinity clustering of document term matrix."""
import json
import os
import pdb
import sys
import numpy as np
from sklearn.cluster import AffinityPropagation as AF
import collections
from scipy.spatial.distance import pdist

# Add current path to python path
sys.path.append(os.getcwd())
import constants


# Read the input data
input_directory = os.getcwd() + '/' + constants.OUTPUT_FOLDER + \
    'Anglo_Saxon_Chronicles/'

# Post-process index data here for the visualization (shorten entries)
with open(input_directory + 'index_with_selected_vocab.json') as json_file:
    index_data = json.load(json_file)
new_index_data = []
for element in index_data:
    if len(element.split('<br>')) > 30:
        new_index_data.append('<br>'.join(element.split('<br>')[0:30]))
    else:
        new_index_data.append(element)

# Load the document term matrix
matrix_documents = np.loadtxt(input_directory + 'document_term_matrix_with_selected_vocab.txt')
# Load all features
with open(input_directory + 'features_with_selected_vocab.json') as json_file:
    features = json.load(json_file)

pdb.set_trace()

# Cluster entries with the help of affinity propagation
modelc = AF(affinity='euclidean', convergence_iter=15, max_iter=1000)
labels = modelc.fit_predict(matrix_documents)
centers = modelc.cluster_centers_indices_.tolist()

# Save the results

with open(input_directory + 'labels_ap_clustering_selected_features.json', 'w') as file:
    file.write(json.dumps(labels.tolist()))

with open(input_directory + 'centers_ap_clustering_selected_features.json', 'w') as file:
    file.write(json.dumps(centers))

def get_top_keywords(data, clusters, labels):
    cluster_ids = set(clusters.tolist())
    for i in cluster_ids:
        common_terms = []
        documents = np.where(clusters == i)
        for document in documents[0].tolist():
            doc_specific_features = np.where(data[document] > 0)
            common_terms.extend(doc_specific_features[0].tolist())
        terms = [labels[element] for element in common_terms]
        common_terms = [item for item, count in
                        collections.Counter(terms).items()
                        if count == len(documents[0].tolist())]
        print('Cluster ' + str(i) + ':\n')
        print('All documents:\n')
        print(' '.join(common_terms))
        print('\n')
        print('At least two third of docs:')

        number = int(len(documents[0].tolist()) * 0.66)
        common_terms = [item for item, count in
                        collections.Counter(terms).items() if count > number]
        print(' '.join(common_terms))
        print('\n')
        total = len(np.where(clusters==i)[0])
        print('Total number of documents in the cluster:')
        print(total)
        print('-'*30)



get_top_keywords(matrix_documents, labels, features)
print("Use the following commands to print information:\n")
print("np.where(labels==1) this prints all document index the cluster number\
 of which is one.\n")

print("index_data[1] this prints the text corresponding to the first document\n")
pdb.set_trace()
