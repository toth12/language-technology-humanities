import os
import sys
import json
import glob
from collections import Counter
import pdb
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from nltk.corpus import brown
import nltk
from nltk import FreqDist


np.set_printoptions(suppress=True)

# Add current path to python path
sys.path.append(os.getcwd())
import constants
from scipy.stats import chisquare

def chi_test(row,total_word_count_corpus = 66859 ,total_word_count_ref_corpus = 31664):
    observed_raw_frequencies_corpus = row['raw_frequency_moonstone']
    observed_raw_frequencies_refcorpus = row['raw_frequency_refcorpus']
    observed_raw_frequencies =np.array([[observed_raw_frequencies_corpus,total_word_count_corpus - observed_raw_frequencies_corpus],
       [observed_raw_frequencies_refcorpus, total_word_count_ref_corpus- observed_raw_frequencies_refcorpus] 
       ])

    chi = chi2_contingency(observed_raw_frequencies.T)
    expected = chi[3].T
    significance = chi[1]<0.05
    component = np.power((observed_raw_frequencies - expected),2)/expected
    strength = (observed_raw_frequencies - expected)[0][0]
    reference_corpus = component[1][0]
    corpus = component[0][0]

    return corpus,reference_corpus,significance,strength

print("-" * 80 + "\n")
print("Identifying the characteristic vocab with filtering began")
print("\n")

output_directory = os.getcwd() + '/' + constants.OUTPUT_FOLDER + \
    'Moonstone/'

input_directory_moonstone = os.getcwd() + '/' + constants.OUTPUT_FOLDER + \
    'Moonstone/'

input_directory_victorian_novels = os.getcwd() + '/' + constants.OUTPUT_FOLDER + \
    'VictorianNovels/'

vocab_moonstone = input_directory_moonstone + 'frequency_all_tokens_without_stopwords.csv'
vocab_victorian = input_directory_victorian_novels + 'frequency_all_tokens_no_stopwords_wuthering_hights.csv'



df_moonstone = pd.read_csv(vocab_moonstone)

df_moonstone = df_moonstone.loc[df_moonstone.raw_frequency >5]

df_victorian = pd.read_csv(vocab_victorian)

df_victorian = df_victorian.loc[df_victorian.raw_frequency >5]

# merge the two tables


df_joint =df_moonstone.merge(df_victorian,on='token',how='inner',suffixes=['_moonstone','_refcorpus'])

df_joint = df_joint.loc[:, ~df_joint.columns.str.contains('^Unnamed')]

total_word_count_moonstone = df_joint.raw_frequency_moonstone.sum()
total_word_count_ref_corpus = df_joint.raw_frequency_refcorpus.sum()
pdb.set_trace()

#input = df_joint.loc[(df_joint['token'] == 'dreadful')][['raw_frequency_moonstone','raw_frequency_refcorpus']].values.tolist()

#chi_result = chi_test(input[0][0],input[0][1],total_word_count_moonstone,total_word_count_ref_corpus)
#df_joint['strength_moonstone'],df_joint['strength_refcorpus'],df_joint['significant'],df_joint['degree_of_strength']  = chi_test(df_joint['raw_frequency_moonstone'], df_joint['raw_frequency_refcorpus'],total_word_count_corpus=total_word_count_moonstone, total_word_count_ref_corpus = total_word_count_ref_corpus)
# df_joint['strength_moonstone'],df_joint['strength_refcorpus'],df_joint['significant'],df_joint['degree_of_strength'] 
result = df_joint.apply(chi_test,axis=1)
result = pd.DataFrame(result.to_list(),columns=['strength_moonstone','strength_refcorpus','significant','ratio_of_strengths'])
df_joint = df_joint.join(result)

df_joint.to_csv(input_directory_moonstone+'moonstone_wuthering_hights_joint.csv')

