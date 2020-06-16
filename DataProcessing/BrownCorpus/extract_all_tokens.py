import os
import spacy
import sys
import json
import glob
import pdb
from nltk.corpus import brown
import nltk


# Add current path to python path
sys.path.append(os.getcwd())
import constants



sp = spacy.load('en_core_web_sm')
sp.max_length = 10000000
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS


print("-" * 80 + "\n")
print("Pre-processing of the Brown Corpus began (takes a few minutes, be patient)")
print("\n")

# Set up the path for the data directory

output_directory = os.getcwd() + '/' + constants.OUTPUT_FOLDER + \
    'BrownCorpus/'

brown_corpus = []
    
[brown_corpus.extend(brown.sents(file_ids)) for file_ids in brown.fileids()]

brown_sentences = []
for element in brown_corpus:
    sentence=' '.join(element)
    brown_sentences.append(sentence)

doc = sp.pipe(brown_sentences,disable=['parser','ner'])

tokens_without_stopwords = []
for element in doc:


# Tokenize the text, filter punctuation and transform to lowercase

    tokens_without_stopwords.extend([token.lemma_.lower() for token in element if (
            (token.pos_ != "PUNCT") and (token.pos_ != "SPACE") and 
            (token.lower_ not in spacy_stopwords)
        )])
with open(output_directory + "all_token_without_stopwords.json", 'w') as outfile:
        json.dump(tokens_without_stopwords, outfile)


print("Pre-processing of the Brown corpus finished")
print ("Results saved to the following folder and file")
print (output_directory)
print ("all_token_without_stopwords.json")
