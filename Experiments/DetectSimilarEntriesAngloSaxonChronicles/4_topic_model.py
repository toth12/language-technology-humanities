
import os
import json
import pdb
import sys
import pandas as pd
from gensim.models import TfidfModel
import numpy as np
from corextopic import corextopic as ct
from corextopic import vis_topic as vt
import matplotlib.pyplot as plt
import itertools
import shutil
import os
from shutil import copyfile



print("-" * 80 + "\n")
print("The topic modelling (standard, anchored and hiearchical) of the Anglo-Saxon Chronicle began")
print("\n")

# Add current path to python path
sys.path.append(os.getcwd())
from Utils import gensim_utils
import constants


def chunks(l, n):
    """Returns every n elements of a list."""
    l = [element for element in l if len(element) > 0]
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i + n]

# Set up the data directory
data_directory = os.getcwd() + '/' + constants.OUTPUT_FOLDER + \
    'Anglo_Saxon_Chronicles/'

# Read the document collection
with open(data_directory +
          'lemmatized_content_words_only_multi_words_document_collection.json') as json_file:
    document_collection = json.load(json_file)
# Get every three sentences
chunked_bows = []
for document in document_collection:
    for element in chunks(document, 3):

        chunked_bows.append(list(itertools.chain.from_iterable(element)))

preselected_features = pd.read_csv(data_directory +
                                   'Precomputed_data/historically_specific_vocab.csv')['0'].to_list()


# Initialize a gensim dictionary
gensim_dic = gensim_utils.initialize_gensim_dictionary([preselected_features])

# Create a TF-IDF based bag of word representation of each entry

# Build a gensim corpus
gensim_corpus = [gensim_dic.doc2bow(text) for text in chunked_bows]

# Train a tf-idf model and reuse the gensim dic
model = TfidfModel(gensim_corpus)

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
entries_for_remove = []
for element in enumerate(matrix_documents):
    if ((np.nonzero(element[1])[0].shape[0] < 5) or
       (np.nonzero(element[1])[0].shape[0] > 100)):
        entries_for_remove.append(element[0])

matrix_documents = np.delete(matrix_documents, entries_for_remove, 0)

# Remove those entries that were removed from document term matrix
for i in sorted(entries_for_remove, reverse=True):
    del chunked_bows[i]

# Create a list from the features (features_list) so that they can be reused
features_list = [element[1] for element in gensim_dic.iteritems()]

# Transform the document term matrix into a binary matrix
doc_word = np.where(matrix_documents > 0, 1, 0)

print("\n")
print ("The following topics were extracted from the Anglo-Saxon Chronicle")
print("\n")
# Run Corex topic modelling
topic_model = ct.Corex(n_hidden=20, max_iter=200, verbose=False, seed=8)
topic_model.fit(np.matrix(doc_word), words=features_list)

# Print document topic matrix: topic_model.log_p_y_given_x

# Print the key topics
topics = topic_model.get_topics()
topics_to_print = []
for n,topic in enumerate(topics):
    topic_words,_ = zip(*topic)
    topics_words_values = []
    for element in topic:
        topics_words_values.append(element[0] + ' (' +
                                   str(np.round(element[1], decimals=3)) + ')')
    topics_to_print.append(','.join(topics_words_values))
    print('{}: '.format(n) + ','.join(topic_words))


# Print top ten documents of each topic
'''
top_docs = topic_model.get_top_docs(n_docs=10, sort_by='log_prob')
selected_chunked_bows = []
for topic_n, topic_docs in enumerate(top_docs):
    docs, probs = zip(*topic_docs)
    docs = [str(element) for element in docs]
    topic_str = str(topic_n + 1) + ': ' + ','.join(docs)
    doc_bows = []
    for doc in docs:
        doc_bows.append(' '.join(chunked_bows[int(doc)]))
    print(topic_str)
'''

# Create a report on the topic modelling (folder named topic-model-report created automatically)
vt.vis_rep(topic_model, column_label=features_list, prefix='topic-model-report')

# Run a hiearchical topic modelling

# Train a second layer to the topic model
tm_layer2 = ct.Corex(n_hidden=3)

tm_layer2.fit(topic_model.labels)

# Train a third layer to the topic model
tm_layer3 = ct.Corex(n_hidden=1)
tm_layer3.fit(tm_layer2.labels)

vt.vis_hierarchy([topic_model, tm_layer2, tm_layer3], column_label=features_list, max_edges=200, prefix='topic-model-hierarchical-report')


# Train an anchored topic modelling
print ("\n")
print ("The following topics anchored to 'kill' were extracted from the Anglo-Saxon Chronicle")
print ("\n")
topic_model = ct.Corex(n_hidden=20, max_iter=200, verbose=False, seed=8)
topic_model.fit(np.matrix(doc_word), words=features_list,
                anchors=['kill'], anchor_strength=5)

# Render the results

# Print the key topics
topics = topic_model.get_topics()
topics_to_print = []
for n,topic in enumerate(topics):
    topic_words,_ = zip(*topic)
    topics_words_values = []
    for element in topic:
        topics_words_values.append(element[0] +
                                   ' (' + str(np.round(element[1], decimals=3)) + 
                                   ')')
    topics_to_print.append(','.join(topics_words_values))
    print('{}: '.format(n) + ','.join(topic_words))


# Print top ten documents of each topic

'''
top_docs = topic_model.get_top_docs(n_docs=10, sort_by='log_prob')
selected_chunked_bows = []
for topic_n, topic_docs in enumerate(top_docs):
    docs, probs = zip(*topic_docs)
    docs = [str(element) for element in docs]
    topic_str = str(topic_n + 1) + ': ' + ','.join(docs)
    doc_bows = []
    for doc in docs:
        doc_bows.append(' '.join(chunked_bows[int(doc)]))
    print(topic_str)
'''

vt.vis_rep(topic_model, column_label=features_list,
           prefix='topic-model-anchored-report')

# Render a diagram about the topic model
plt.figure(figsize=(10, 5))
plt.bar(range(topic_model.tcs.shape[0]), topic_model.tcs, color='#4e79a7',
        width=0.5)
plt.xlabel('Topic', fontsize=16)
plt.ylabel('Total Correlation (nats)', fontsize=16)
plt.savefig(data_directory+'topics.png', bbox_inches='tight')



# copy output to output data folder
cwd = os.getcwd()
# check if folder already exists

if not os.path.isdir(data_directory+'topic-model-anchored-report'):

    shutil.move(cwd+'/topic-model-anchored-report', data_directory)
    shutil.move(cwd+'/topic-model-hierarchical-report', data_directory)
    shutil.move(cwd+'/topic-model-report', data_directory)

else:

    shutil.rmtree(data_directory+'/topic-model-anchored-report')
    shutil.rmtree(data_directory+'/topic-model-report')
    shutil.rmtree(data_directory+'/topic-model-hierarchical-report')


    shutil.move(cwd+'/topic-model-anchored-report', data_directory)
    shutil.move(cwd+'/topic-model-hierarchical-report', data_directory)
    shutil.move(cwd+'/topic-model-report', data_directory)

print ("Topic modelling finished, reports were copied to the following folders:")
print ("\n")
print (data_directory+'topic-model-anchored-report')
print (data_directory+'topic-model-report')
print (data_directory+'topic-model-hierarchical-report')



