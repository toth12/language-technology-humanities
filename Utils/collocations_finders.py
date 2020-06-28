__author__ = 'roysy'
from nltk.collocations import *
from nltk.collocations import BigramAssocMeasures, TrigramAssocMeasures
from nltk.probability import *
from nltk.probability import FreqDist as fd
import pdb
import nltk
from nltk.probability import MLEProbDist
import numpy as np

def bigram_collocation_finder(tokens, window_size=2):
    '''It returns bigram collocations, including their raw frequency, by using a list of tokens as input. Window size is two by defeault
     Parameters
    -----------
    tokens: a list of tokens or list of sentences that are list of tokens
    window_size: the window size of the collocation, by default 3

    Returns
    -------
    bigram_collocations: list of bigram collocations and their raw frequency in tuples

    '''


    bigram_measures = BigramAssocMeasures()
    if isinstance(tokens[0],list):
        finder=BigramCollocationFinder.from_words(BigramCollocationFinder._build_new_documents(tokens, window_size, pad_right=True),window_size=window_size)

        
    else:
        finder = BigramCollocationFinder.from_words(tokens,window_size=window_size)

    result = finder.score_ngrams(bigram_measures.raw_freq)
    return result

def bigram_collocation_finder_with_pointwise_mutual_information(tokens,window_size=2):
    '''It returns bigram collocations, including their pointwise mutual information, by using a list of tokens or list of sentences that are list of tokens as input. Window size is two.
     Parameters
    -----------
    tokens: a list of tokens or list of sentences that are list of tokens
   window_size: the window size of the collocation, by default 3


    Returns
    -------
    bigram_collocations: list of bigram collocations and their raw frequency in tuples

    '''

    bigram_measures = BigramAssocMeasures()
    if isinstance(tokens[0],list):
        finder=BigramCollocationFinder.from_words(BigramCollocationFinder._build_new_documents(tokens, window_size, pad_right=True),window_size=window_size)

    else:
        finder = BigramCollocationFinder.from_words(tokens,window_size=window_size)


    result = finder.score_ngrams(bigram_measures.pmi)
    return result


def bigram_collocation_finder_with_log_likelihood_ratio(tokens,window_size=2):
    '''It returns bigram collocations, including their pointwise mutual information, by using a list of tokens or list of sentences that are list of tokens as input. Window size is two.
     Parameters
    -----------
    tokens: a list of tokens or list of sentences that are list of tokens
   window_size: the window size of the collocation, by default 3


    Returns
    -------
    bigram_collocations: list of bigram collocations and their raw frequency in tuples

    '''

    bigram_measures = BigramAssocMeasures()
    if isinstance(tokens[0],list):
        
        finder=BigramCollocationFinder.from_words(BigramCollocationFinder._build_new_documents(tokens, window_size, pad_right=True),window_size=window_size)
        
        #this is the original code
        #finder = BigramCollocationFinder.from_documents(tokens)
    else:
        finder = BigramCollocationFinder.from_words(tokens,window_size=window_size)

  
    result = finder.score_ngrams(bigram_measures.likelihood_ratio)
    return result






def trigram_collocation_finder(tokens,window_size = 3):
    '''It returns trigram collocations, including their raw frequency, by using a list of tokens or list of sentences that are list of tokens as input. Window size is three.
     Parameters
    -----------
   tokens: a list of tokens or list of sentences that are list of tokens
   window_size: the window size of the collocation, by default 3

    Returns
    -------
    bigram_collocations: list of bigram collocations and their raw frequency in tuples

    '''

    trigram_measures = TrigramAssocMeasures()
    if isinstance(tokens[0],list):
        # todo how to measure the window size here
        finder = TrigramCollocationFinder.from_documents(tokens)
    else:
        finder = TrigramCollocationFinder.from_words(tokens,window_size=window_size)


    result = finder.score_ngrams(trigram_measures.raw_freq)
    return result



def trigram_collocation_finder_with_pointwise_mutual_information(tokens,window_size = 3):
    '''It returns bigram collocations, including their pointwise mutual information, by using a list of tokens or list of sentences that list of tokens as input. Window size is two.
     Parameters
    -----------
    tokens: a list of tokens or list of sentences that are list of tokens
    window_size: the window size of the collocation, by default 3


    Returns
    -------
    bigram_collocations: list of bigram collocations and their raw frequency in tuples

    '''

    trigram_measures = TrigramAssocMeasures()
    if isinstance(tokens[0],list):
        # todo how to measure the window size here
        finder = TrigramCollocationFinder.from_documents(tokens)
    else:
        finder = TrigramCollocationFinder.from_words(tokens,window_size=window_size)

    result = finder.score_ngrams(trigram_measures.pmi)
    return result



def bigram_collocation_finder_with_count(tokens,window_size=2):
    '''It returns bigram collocations, including their count, by using a list of tokens or list of sentences that are list of tokens as input. Window size is two.
     Parameters
    -----------
    tokens: a list of tokens or list of sentences that are list of tokens
   window_size: the window size of the collocation, by default 3


    Returns
    -------
    bigram_collocations: list of bigram collocations and their raw frequency in tuples

    '''

    if isinstance(tokens[0],list):
   
        finder=BigramCollocationFinder.from_words(BigramCollocationFinder._build_new_documents(tokens, window_size, pad_right=True),window_size=10)
        
        #this is the original code
        #finder = finder.from_documents(tokens)
        #pdb.set_trace()
    else:
        finder = BigramCollocationFinder.from_words(tokens,window_size=window_size)

   
    result=[]
   

    for k,v in finder.ngram_fd.items():
        result.append((k,v))
    return result


if __name__ == "__main__":
    token_list = [['Hi','how','you','today'],['you','are','nice']]
    print (token_list)
    result = bigram_collocation_finder_with_count(token_list,window_size=10)
    