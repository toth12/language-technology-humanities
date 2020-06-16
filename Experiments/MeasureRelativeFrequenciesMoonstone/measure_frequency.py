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


print("-" * 80 + "\n")
print("Measuring of word frequencies in the Moonstone by Wilkie Collins began")
print("\n")

output_directory = os.getcwd() + '/' + constants.OUTPUT_FOLDER + \
    'Moonstone/'

def calculate_relative_freq(raw_frequency,total_word_count):
    result = (raw_frequency/total_word_count)*1000
    result_formatted = '{0:.10f}'.format(result)
    return result_formatted

input_directory = os.getcwd() + '/' + constants.OUTPUT_FOLDER + \
    'Moonstone/'

# Get all file names in that directory
input_files = glob.glob(input_directory + '*.json')

for element in input_files:
    with open(element) as json_file:
        tokens = json.load(json_file)
        
        file_name = element.split('_')
    
    count_of_tokens = Counter(tokens)
    df = pd.DataFrame(list(count_of_tokens.items()),columns=['token','raw_frequency'])
    df['relative_frequency_per_thousand_words'] = df['raw_frequency'].apply(calculate_relative_freq,total_word_count=len(tokens))
    df = df.sort_values("raw_frequency",ascending=False).reset_index()
    df['ranking']=df.index+1
    del df['index']
    if 'with' not in file_name:
        index_dreadful = df[df.token=="dreadful"].index[0]
        print ("The relative frequency per thousand word of dreadful is: ")
        print(df.iloc()[index_dreadful]['relative_frequency_per_thousand_words'])
        print ("The raw frequency (count) of dreadful is: ")
        print(df.iloc()[index_dreadful]['raw_frequency'])
        print ("The position of dreadful in the frequency ranking:")
        print(df.iloc()[index_dreadful]['ranking'])
        print ('\n')
    if 'with' in file_name:
        print("Total word count (with stopwords) in the Moonstone: " + str(len(tokens)))
        df.to_csv(output_directory+'frequency_all_tokens_with_stopwords.csv')
        print ('\n')
    else:
        print("Total word count (without stopwords) in the Moonstone: "  + str(len(tokens)))
        df.to_csv(output_directory+'frequency_all_tokens_without_stopwords.csv')
        print ('\n')


print("Measuring of word frequencies in the Moonstone by Wilkie Collins finished")
print ("Results are saved to the following folder and files")

print (output_directory)
print ("frequency_all_tokens_with_stopwords.csv")
print ("frequency_all_tokens_without_stopwords.csv")

