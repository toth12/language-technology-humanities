"""Creates an affinity propagation clustering of document term matrix.

As input, it takes the document term matrix, features list, and the
document collection. As output, it returns the cluster membership of
each entry in the document collection in a numpy array saved into
Data/Outputs/Anglo_Saxon_Chronicles/clusters.json. It also prints keywords
connecting members of each cluster. Connecting keywords are those features
that occur at least three third of cluster members.
"""
import json
import os
import pdb
import sys
import numpy as np
from sklearn.cluster import AffinityPropagation as AF
import collections


print("-" * 80 + "\n")
print("Clustering the annals of the Anglo Saxon Chronicle began")
print("\n")

# Add current path to python path
sys.path.append(os.getcwd())
import constants

# Set the input directory
data_directory = os.getcwd() + '/' + constants.OUTPUT_FOLDER + \
    'Anglo_Saxon_Chronicles/'

# Load the post processed document collection
input_file = 'post_processed_document_collection.json'
with open(data_directory + input_file) as json_file:
    document_collection = json.load(json_file)

# Load the features
with open(data_directory +
          'features_list_reshaped_TF_IDF_training.json') as json_file:
    features = json.load(json_file)


# Load the document term matrix
documents_terms_matrix_1 = np.loadtxt(data_directory +
                                    'document_term_matrix_TF_IDF.txt')



# Cluster entries with the help of affinity propagation
modelc = AF(affinity='euclidean', convergence_iter=15, max_iter=1000)
labels = modelc.fit_predict(documents_terms_matrix_1)
centers = modelc.cluster_centers_indices_.tolist()


def get_top_keywords(data, clusters, labels):
    """Find those features that occur at least two third of a given
    cluster's members and prints them, as well as total number of members.

    Parameters
    ----------
    data : {numpy array}
        document term matrix
    clusters : {numpy array}
        cluster labels
    labels : {list of strings}
        list of features
    """
    report = ''
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
        report +='\nCluster ' + str(i) + ':\n'
        if len(common_terms) >0:
            report +='All documents in the cluster has the following keywords:\n'
            report +=' '.join(common_terms)
        report +='\n'
        report +='At least two third of docs have the following keywords in the cluster:\n'

        number = int(len(documents[0].tolist()) * 0.66)
        common_terms = [item for item, count in
                        collections.Counter(terms).items() if count > number]
        report +=' '.join(common_terms)
        report +='\n'
        total = len(np.where(clusters == i)[0])
        report +='Total number of documents in the cluster: '
        report +=str(total)
        report +='\n'
        report +='-' * 30
    return report


# Save the clusters
with open(data_directory + 'clusters.json', 'w') as file:
    file.write(json.dumps(labels.tolist()))

# Save the cluster report
report = get_top_keywords(documents_terms_matrix_1, labels, features)
with open(data_directory + 'cluster_report.txt', 'w') as file:
    file.write(report)
print("Clustering the annals of the Anglo Saxon Chronicle finished")
print ("In total "+str(len(set(labels)))+ " clusters were detected")

print ("A report about the clusters (keywords connecting cluster members, and the number of documents belonging to the cluster) has been written to the following file in the following directory")
print (data_directory)
print ("cluster_report.txt")


'''
get_top_keywords(documents_terms_matrix_1, labels, features)
print("Use the following commands to print information:\n")
print("np.where(labels==1) this prints all document index the cluster number\
 of which is one.\n")

print("document_collection[1] this prints the text corresponding to\
the first document\n")
'''

