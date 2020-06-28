import constants
import os
import json
import pdb
from Utils import collocations_finders as cf
import numpy as np


def run():
    
    input_directory = os.getcwd() + '/' + constants.OUTPUT_FOLDER + \
    'Moonstone/'


    input_file = 'moonstone_all_tokens_in_sentences_with_stopwords.json'


    with open(input_directory+input_file) as json_file:
        data = json.load(json_file)
        
    
    # Print the total word count
    tokens = []
    for sentence in data:

        [tokens.append(token) for token in sentence]


    print ("The total word count of The Moonstone without stopwords")
    print (len(tokens))
    bigrams = cf.bigram_collocation_finder_with_count(data,window_size=10)
    collocations_limping = [element for element in bigrams if element[0][0]=="limping" or element[0][1]=="limping"]
    collocations_lucy = [element for element in bigrams if element[0][0]=="lucy" or element[0][1]=="lucy"]

    count_limping_lucy = [element for element in bigrams if element[0][0]=="limping" and element[0][1]=="lucy"][0][1]
    count_lucy_lame = [element for element in bigrams if element[0][0]=="lucy" and element[0][1]=="lame"][0][1]

    print ("The count (raw frequency) of limping Lucy")
    print (count_limping_lucy)

    count_be_lucy = [element for element in bigrams if element[0][0]=="be" and element[0][1]=="lucy"][0][1]
    

    print ("The count (raw frequency) of be Lucy")
    print (count_be_lucy)

    print ("The count (raw frequency) of Lucy lame")

    print(count_lucy_lame)

    print ("The relative frequency or observed probability (per 10000 words) of limping Lucy")
    print (count_limping_lucy/len(tokens)*10000)

    observed_p_limping_lucy= count_limping_lucy/len(tokens)

    print ("The relative frequency or observed probability (per 10000 words) of be Lucy")
    print (count_be_lucy/len(tokens)*10000)

    print ("The count of Lucy")
    print (tokens.count('lucy'))

    print ("The relative frequency (per 1000 words) of Lucy")
    print (tokens.count('lucy')/len(tokens)*1000)

    p_lucy=tokens.count('lucy')/len(tokens)

    print ("The count of be")
    print (tokens.count('be'))

    print ("The count of lame")
    print (tokens.count('lame'))


    print ("The relative frequency of be")
    print (tokens.count('be')/len(tokens))

    print ("The count of limping")
    print (tokens.count('limping'))

    print ("The relative frequency (per 10000 words) of limping")
    print (tokens.count('limping')/len(tokens)*10000)

    p_limping = tokens.count('limping')/len(tokens)

    print ("The theoretical probability (per 100000000 words) for limping Lucy")

    print ((tokens.count('limping')/len(tokens))*(tokens.count('lucy')/len(tokens))*100000000)

    print ("The theoretical probability (per 100000 words) for be Lucy")

    print ((tokens.count('be')/len(tokens))*(tokens.count('lucy')/len(tokens))*100000)

    pmi_limping_lucy = np.log2(np.true_divide((count_limping_lucy/len(tokens)), (tokens.count('limping')/len(tokens))*(tokens.count('lucy')/len(tokens))))
    pmi_be_lucy = np.log2(np.true_divide((count_be_lucy/len(tokens)), (tokens.count('be')/len(tokens))*(tokens.count('lucy')/len(tokens))))
    pmi_lucy_lame = np.log2(np.true_divide((count_lucy_lame/len(tokens)), (tokens.count('lucy')/len(tokens))*(tokens.count('lame')/len(tokens))))

    print ("Pmi of limping Lucy")
    print (pmi_limping_lucy)

    print ("Pmi of be Lucy")
    print (pmi_be_lucy)

    print ("Pmi of Lucy lame")
    print (pmi_lucy_lame)





if __name__ == "__main__":

    run()

