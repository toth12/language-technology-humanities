import os
import sys
import json
import glob
from collections import Counter
import pdb
import pandas as pd

# Add current path to python path
sys.path.append(os.getcwd())
import constants

def calculate_relative_freq(raw_frequency,total_word_count):
    result = (raw_frequency/total_word_count)*1000
    result_formatted = '{0:.10f}'.format(result)
    return result_formatted

print("-" * 80 + "\n")
print("Measuring of word frequencies in the Victorian novels mini corpus began")
print("\n")

output_directory = os.getcwd() + '/' + constants.OUTPUT_FOLDER + \
    'VictorianNovels/'

input_directory = os.getcwd() + '/' + constants.OUTPUT_FOLDER + \
    'VictorianNovels/'

# Get all file names in that directory
input_files = glob.glob(input_directory + '*.json')

all_tokens = []
for element in input_files:
    with open(element) as json_file:
        tokens = json.load(json_file)

        all_tokens.extend(tokens)


count_of_tokens = Counter(all_tokens)
df = pd.DataFrame(list(count_of_tokens.items()),columns=['token','raw_frequency'])
df['relative_frequency_per_thousand_words'] = df['raw_frequency'].apply(calculate_relative_freq,total_word_count=len(all_tokens))

df.sort_values(by=['raw_frequency'],ascending=False).to_csv(output_directory+'frequency_all_tokens_no_stopwords.csv')

print("Measuring of word frequencies in the Victorian novels mini corpus finished")
print ("Results saved to the following folder and file")
print (output_directory)
print ("frequency_all_tokens_no_stopwords.csv")
