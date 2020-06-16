"""Creates a distance matrix of certain entries of the document term matrix.

The distance matrix is rendered as a heatmap.
"""


import os
import pdb
import sys
import numpy as np
import plotly
from sklearn.metrics.pairwise import cosine_distances
import plotly.figure_factory as ff
from plotly.graph_objs import *
from sklearn.metrics.pairwise import cosine_similarity

print("-" * 80 + "\n")
print("Measuring the Cosine Similarity Between Annals of the Anglo-Saxon Chronicle, and the visualization through heatmap began")
print("\n")




# Add current path to python path
sys.path.append(os.getcwd())
import constants


# Set the input directory
input_directory = os.getcwd() + '/' + constants.OUTPUT_FOLDER + \
    'Anglo_Saxon_Chronicles/'

row_numbers = ['101', '93', '95', '6', '94']

# Load the document term matrix
matrix_documents = np.loadtxt(input_directory +
                              'document_term_matrix_TF_IDF.txt')

# Get the rows corresponding to the selected documents

matrix_documents_selected = []
for f in row_numbers:
    matrix_documents_selected.append(matrix_documents[int(f)])

matrix_documents_selected = np.vstack(matrix_documents_selected)


similarities = cosine_similarity(matrix_documents_selected)
similarities = np.around(similarities, decimals=2)

print ("The similarity matrix of entries in Group 2")
print('pairwise dense output:\n {}\n'.format(similarities))

row_names = []
for element in row_numbers:
    row_names.append("Row N. " + element)

colorscale = [[0, 'navy'], [1, 'plum']]
font_colors = ['white', 'black']

fig = ff.create_annotated_heatmap(z=similarities.tolist(), x=row_names,
                                  y=row_names,
                                  annotation_text=similarities.tolist(),
                                  colorscale=colorscale,
                                  font_colors=font_colors)
plotly.offline.plot(fig, filename=input_directory + 'cluster_heatmap.html')

print ("The heatmap is available in the following directory and file:\n")
print (input_directory)
print ("cluster_heatmap.html")

