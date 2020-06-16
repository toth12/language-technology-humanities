"""Removes all terms (lemmas that are content words), df of which is below 5."""
import os
import json
import pdb
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing

# Add current path to python path
sys.path.append(os.getcwd())
from Utils import gensim_utils
import constants
from gensim.models import TfidfModel


print("-" * 80 + "\n")
print("Elimination of terms the document frequency of which is below 5 is beginning.")
print("\n")
# Set up the path for the data directory
data_directory = os.getcwd() + '/' + constants.OUTPUT_FOLDER + \
    'Anglo_Saxon_Chronicles/'
input_file_name = "lemmatized_content_words_only_multi_words_document_collection.json"
output_file_name = "features_selected_with_unsupervised_methods.csv"
# Read the document collection bows of lemmatized content words
with open(data_directory + input_file_name) as json_file:
    document_collection = json.load(json_file)

# Measure the document frequency of each term with the help of gensim
new_document_collection = []
for entry in document_collection:
    # Each document has to be represented as a list of terms and not list of sentences
    # Get all terms in an entry
    all_terms = []
    for sentence in entry:
        for term in sentence:
            all_terms.append(term)
    new_document_collection.append(all_terms)


# Initialize a gensim dictionary with the new document collection
gensim_dic = gensim_utils.initialize_gensim_dictionary(new_document_collection)

# Get the document frequency of each term in a panda dataframe
dfs = gensim_utils.get_df_in_dictionary(gensim_dic, as_pandas_df=True)



print(str(len(dfs[dfs[1] == 1]) / len(dfs) * 100) + '% of all terms occur only in one document')

# Eliminate those terms that occur in less five documents
gensim_dic.filter_extremes(no_below=5)

# Write the remaining terms out to a file
dfs = gensim_utils.get_df_in_dictionary(gensim_dic, as_pandas_df=True)
print("After eliminating terms the document frequency of which is less than five, " + str(len(dfs)) + " terms are remaining.")


# Train a tf-idf model to evaluate the importance of features

# Build a gensim corpus
gensim_corpus = [gensim_dic.doc2bow(text) for text in new_document_collection]
model = TfidfModel(gensim_corpus, dictionary=gensim_dic)

# Build a tf-idf corpus
corpus_tfidf = model[gensim_corpus]

# Build the matrix representation (matrix_documents) of the TF-IDF corpus
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

# Find the most important features with a random classifier

print("-" * 80 + "\n")
print("Assessment of term importance with RandomForestClassifier began")
print("\n")


clf = RandomForestClassifier()
lab_enc = preprocessing.LabelEncoder()
training_scores_encoded = np.array(list(range(0, matrix_documents.shape[0])))
clf.fit(matrix_documents, training_scores_encoded)
features_importance = np.array(clf.feature_importances_)

# Export the feature importance to panda dataframe and then export it to CSV
feature_list = gensim_utils.get_df_in_dictionary(gensim_dic,True)
feature_list['feature_importance'] = features_importance.tolist()

feature_list.to_csv(data_directory + output_file_name)

print("-" * 80 + "\n")
print("Assessment of term importance with RandomForestClassifier finished\n")
print("\n")

print("List of terms the document frequency of which is more than\
 four (including  their document frequency and their importance assessed with \
 RandomForestClassifier) is saved into the following folder and file:")
print (data_directory)
print (output_file_name)


print("\n")
print("Accomplish the human supervision of this file")
print("Select those terms from this list that are significant for general historical patterns")
print("\n")
print ("Save the result of human selection into the following folder and file")
print (data_directory)
print ("features_selected_with_supervised_methods.csv")

