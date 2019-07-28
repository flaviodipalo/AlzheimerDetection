from feature_sets.psycholinguistic import get_psycholinguistic_features
import nltk
from feature_sets.psycholinguistic import get_psycholinguistic_features
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tqdm import tqdm
import pandas as pd
sid = SentimentIntensityAnalyzer()


data = pd.read_pickle('data/pitt_full_interview.pickle')
anagraphic_data = pd.read_pickle('data/anagraphic_dataframe.pickle')
merged_dataframe = pd.merge(data, anagraphic_data, on='id')


new_dataframe = []
for index, row in tqdm(merged_dataframe.iterrows()):
    single_sentence_list = []

    string = ''
    for token in row.text:
        if token == '\n':
            single_sentence_list.append(string)
            string = ''
        else:
            string += ' ' + token

    counter = 0
    comp_sentiment_sum = 0
    for sentence in single_sentence_list:
        ss = sid.polarity_scores(sentence)
        comp_sentiment_sum += ss['compound']
        counter += 1

    if counter != 0:
        average_sentiment = comp_sentiment_sum / counter
    else:
        average_sentiment = 0

        ## for each interview in the dataset.
    interview = nltk.pos_tag(row.text, lang='eng')

    final_interview = []
    for uttr in interview:
        final_interview.append({'token': uttr[0], 'pos': uttr[1]})

    dict = get_psycholinguistic_features(final_interview)

    dict['average_sentiment'] = average_sentiment

    additional_features = []

    for key, value in dict.items():
        additional_features.append(value)

    ##Here we take in consideration anagraphic features.

    anagraphic_features = [row.age, row.education, row.race, row.sex]

    dict['features'] = additional_features + anagraphic_features
    dict['label'] = row.label
    dict['text'] = row.text

    new_dataframe.append(dict)

final_dataframe = pd.DataFrame(new_dataframe)
import pickle
with open('data/pitt_full_interview_features.pickle', 'wb') as f:
    pickle.dump(final_dataframe, f)
