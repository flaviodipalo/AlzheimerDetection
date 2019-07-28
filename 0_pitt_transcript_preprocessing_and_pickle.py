from __future__ import with_statement
import os
import re
import pandas as pd
import pickle
import nltk
from nltk.tokenize import sent_tokenize,word_tokenize

GET_INV= False

def file_tokenization(input_file):
    '''
    :param input_file: single dataset file as readed by Python
    :return: tokenized string of a single patient interview
    '''
    output_list = []
    id_string = input_file.name.split('/')[-1]
    print(id_string)
    result = re.search('(.*).cha',id_string)
    id = result.group(1)
    for line in input_file:
        for element in line.split("\n"):
            if "*PAR" in element or ("*INV" in element and GET_INV):
                #remove any word after the period.
                cleaned_string = element.split('.', 1)[0]
                #replace par with empty string, deleting the part of the string that starts with PAR
                cleaned_string = re.sub(r'[^\w]+', ' ', cleaned_string.replace('*PAR',''))
                #substitute numerical digits, deleting underscores
                cleaned_string = re.sub(r'[\d]+','',cleaned_string.replace('_',''))
                tokenized_list = word_tokenize(cleaned_string)
                output_list = output_list+['\n'] + tokenized_list
    return output_list,id

def extact_age_and_sex(file):
    print(file)

def generate_full_interview_dataframe():
    """
    generates the pandas dataframe containing for each interview its label.
    :return: pandas dataframe.
    """
    dementia_list = []
    for label in ["Control", "Dementia"]:
        if label == "Dementia":
            folders = ["cookie", "fluency", "recall", "sentence"]
        else:
            folders = ["cookie"]

        for folder in folders:
            PATH = "data/Pitt_transcripts/" + label + "/" + folder
            for path, dirs, files in os.walk(PATH):
                for filename in files:
                    fullpath = os.path.join(path, filename)
                    with open(fullpath, 'r')as input_file:
                        tokenized_list,id = file_tokenization(input_file)
                        dementia_list.append(
                            {'text':tokenized_list,
                             'label':label,
                             'id':id}
                            )
    dementia_dataframe = pd.DataFrame(dementia_list)
    return dementia_dataframe

dataframe = generate_full_interview_dataframe()
print(dataframe.head())
with open('data/pitt_full_interview.pickle', 'wb') as f:
    pickle.dump(dataframe, f)

