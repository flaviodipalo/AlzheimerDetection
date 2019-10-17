# Enriching Neural Model with Targeted Features for Dementia Detection #

The implementation details of our model are contained in the file: 4_final_model.py.
Here we are providing the latest version of the CNN-LSTM model including:

- hand-crafted features
- the attention mechanism
- class weights balance


## Download Pitt Corpus ##

In order to run the experiments it is necessary to download the Pitt Corpus transcripts from here:
https://dementia.talkbank.org/access/English/Pitt.html

We do not include that data in our submission because it is private, and authorization to use them is needed.  To obtain authorization, follow the instructions at: https://dementia.talkbank.org/.

## Folder Structure ##

Once the data has been downloaded it is necessary to maintain this folder structure; empty folders are left in our supplement as a reference:


```
root:
 - data:
 -- Pitt_transcripts:
 --- Control:
 ---- cookie:
 ---- fluency:
 --- Dementia:
 ---- cookie:
 ---- fluency:
 ---- recall:
 ---- sentence:
```
## Instructions ##

It is necessary to run the following steps:

0) Run file 0_pitt_transcript_preprocessing_and_pickle.py.  This will preprocess the interviews and create a .pickle file.
1) Run 1_pitt_anagraphic_information.py.  This script will produce a .pickle file containing demographic information for the patients starting from the file anagraphic_modded.csv (this section of the dataset is freely available).
2) Run 2_psycolinguistic_features_computation_and_merge.py.  This file will merge the above produced files and compute other linguistic features mentioned in the paper. This file will produce "pitt_full_interview_features.pickle," which is necessary to run the model.

3) Download Glove embeddings 300d from: http://nlp.stanford.edu/data/glove.6B.zip and place them into the glove.6B folder.

4) Run the  4_final_model.py file.  This file will train the model and perform tests with three different data shuffles.  It will produce a list of three dictionaries
containing fundamental classifier metrics obtained on each split.
