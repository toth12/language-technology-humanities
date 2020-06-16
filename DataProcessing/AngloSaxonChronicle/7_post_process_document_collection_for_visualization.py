"""Creates a post-processed document collection for visualization.

During the TF-IDF training process, outlier entries were removed; by
taking their index, these entries are removed from the document
collection. The post-processed document collection
also contains the list of features used during the training process.
"""
import json
import os
import sys
import numpy as np
import pdb
import re

# Add current path to python path
sys.path.append(os.getcwd())
import constants

print("-" * 80 + "\n")
print("Post-processing of document collection began.")
print("\n")

data_directory = os.getcwd() + '/' + constants.OUTPUT_FOLDER + \
    'Anglo_Saxon_Chronicles/'
input_file_name = "document_collection.json"

# Load the document collection
with open(data_directory + input_file_name) as json_file:
    document_collection = json.load(json_file)

# Load the index of entries that were removed
with open(data_directory + 'Precomputed_data/removed_entries.json') as json_file:
    entries_for_remove = json.load(json_file)

# Load the features list
with open(data_directory +
          'features_list_reshaped_TF_IDF_training.json') as json_file:
    features = json.load(json_file)

# Load the document term matrix
documents_terms_matrix = np.loadtxt(data_directory +
                                    'document_term_matrix_TF_IDF.txt')


# Remove those entries that were removed from document term matrix
for i in sorted(entries_for_remove, reverse=True):
    del document_collection[i]

print("-" * 80 + "\n")
print(str(len(entries_for_remove)) + " entries were removed from the \
document collection")
print("\n")

# Replace line breaks with <br> element
document_collection = ['<br> '.join(element.split('\n'))
                       for element in document_collection]
# Remove too long spaces
document_collection = [re.sub(' +', ' ', element) for element
                       in document_collection]

# Add the list of features to each entry of the document collection

post_processed_document_collection = []

for i, entry in enumerate(document_collection):
    # Find the non zero element in each row of the document term matrix
    vocab = np.nonzero(documents_terms_matrix[i])[0].tolist()
    # Add them to the index data
    for element in vocab:
        entry = entry + '<br>' + features[element]
    post_processed_document_collection.append(entry)

# Save the post-processed document collection
output_file = 'post_processed_document_collection.json'
with open(data_directory + output_file, 'w') as file:
    file.write(json.dumps(post_processed_document_collection))

print("-" * 80 + "\n")
print("Post-processing of document collection finished; results were written \
into" + data_directory + output_file)
print("\n")
