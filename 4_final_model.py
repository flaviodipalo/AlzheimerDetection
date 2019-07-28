import os
import pickle
import nltk
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')

from keras import Input, Model, initializers
from keras.layers import Dense, LSTM,Flatten, Dropout, Conv1D, MaxPooling1D, concatenate,Bidirectional, Layer
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping,TensorBoard
from keras.optimizers import Adagrad
from keras import backend as K
from sklearn.metrics import confusion_matrix,precision_score,recall_score

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_curve, auc, accuracy_score

### Here we set some parametrs used in the next scenario
vocabulary_size = 30000
sequence_len = 73
EMBEDDING_SIZE = 300

pos_tag_len = len(['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT',
                         'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP',
                         'VBZ', 'WDT', 'WP', 'WP$', 'WRB'])

### Attention Layer, a contribution from: https://github.com/richliao/textClassifier/blob/master/textClassifierHATT.py
class AttLayer(Layer):
    def __init__(self, attention_dim):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim, )))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.trainable_weights = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

## Some utility functions used later
def format_example(wordlist):
    string = ''
    for word in wordlist:
        string = string + ' ' + word
    return string

def binary_conversion(decimal_list:list,threshold:float):
    return_list = []
    for element in decimal_list:
        if element >= threshold:
            return_list.append(1)
        if element < threshold:
            return_list.append(0)
    return return_list


## The following function is used to create the CNN-LSTM model with attention mechanism and a dense layer to take in consideration hand-crafted linguistic features 
## at the end of the preprocessing stage. 

def create_CNN_LSTM_POS_model_attention(vocabulary_size, sequence_len, embedding_matrix, EMBEDDING_SIZE,pos_tag_list_len,len_features):
    max_seq_length = sequence_len
    deep_inputs = Input(shape=(max_seq_length,))

	# Embedding layer using GloVe embeddings 
    embedding = Embedding(vocabulary_size, EMBEDDING_SIZE, input_length=sequence_len, weights=[embedding_matrix],
                          trainable=False)(deep_inputs) 
	
	# Layer considering the POS tags 
    pos_tagging = Input(shape=(pos_tag_list_len,1))
	
	# hand-crafted features 
    other_features = Input(shape=(len_features, 1))

    dense_1 = Dense(16, activation="sigmoid")(other_features)
    dense_2 = Dense(8, activation="sigmoid")(dense_1)
    dense_3 = Dense(4, activation="sigmoid")(dense_2)

    dropout_rate = 0.5

	# Convolutional Neural Network architecture 
	
    def convolution_and_max_pooling(input_layer):
        print(input_layer)
        conv1 = Conv1D(100, (3), activation='relu')(input_layer)
        dropout_1 = Dropout(dropout_rate)(conv1)
        conv2 = Conv1D(100, (4), activation='relu')(input_layer)
        dropout_2 = Dropout(dropout_rate)(conv2)
        conv3 = Conv1D(100, (5), activation='relu')(input_layer)
        dropout_3 = Dropout(dropout_rate)(conv3)
        conv4 = Conv1D(100, (6), activation='relu')(input_layer)
        dropout_4 = Dropout(dropout_rate)(conv4)
        maxpool1 = MaxPooling1D(pool_size=sequence_len-2)(dropout_1)
        maxpool2 = MaxPooling1D(pool_size=sequence_len-3)(dropout_2)
        maxpool3 = MaxPooling1D(pool_size=sequence_len-4)(dropout_3)
        maxpool4 = MaxPooling1D(pool_size=sequence_len-5)(dropout_4)
        return (maxpool1, maxpool2, maxpool3, maxpool4)

    def convolution_and_max_pooling2(input_layer):
        print(input_layer)
        conv1 = Conv1D(100, (3), activation='relu')(input_layer)
        dropout_1 = Dropout(dropout_rate)(conv1)
        conv2 = Conv1D(100, (4), activation='relu')(input_layer)
        dropout_2 = Dropout(dropout_rate)(conv2)
        conv3 = Conv1D(100, (5), activation='relu')(input_layer)
        dropout_3 = Dropout(dropout_rate)(conv3)
        conv4 = Conv1D(100, (6), activation='relu')(input_layer)
        dropout_4 = Dropout(dropout_rate)(conv4)
        maxpool1 = MaxPooling1D(pool_size=33)(dropout_1)
        maxpool2 = MaxPooling1D(pool_size=32)(dropout_2)
        maxpool3 = MaxPooling1D(pool_size=31)(dropout_3)
        maxpool4 = MaxPooling1D(pool_size=30)(dropout_4)
        return (maxpool1, maxpool2, maxpool3, maxpool4)
    max_pool_emb = convolution_and_max_pooling(embedding)
    max_pool_pos = convolution_and_max_pooling2(pos_tagging)

    cc1 = concatenate([max_pool_emb[0], max_pool_emb[1], max_pool_emb[2], max_pool_emb[3],
                       max_pool_pos[0], max_pool_pos[1], max_pool_pos[2], max_pool_pos[3]],
                      axis=2)

	# Bidirectional LSTM and attention layer 

    lstm = Bidirectional(LSTM(300, return_sequences=True))(cc1)
    attention = AttLayer(300)(lstm)
	
	# the results produced by the latest dense layer are flattened and then concatenated with the results produced by the attention layer. 
	
    flat_classifier = Flatten()(dense_3)
    concatenation_layer = concatenate([attention, flat_classifier])
    output = Dense(1, activation="sigmoid")(concatenation_layer)
	
	# Our model will take as input three sequences:  
	# - deep_inputs: the full list of the word spoken by the patient during the interview 
	# - pos_tagging: the pos tagging list relative to the first input 
	# - other_features: the handcrafted linguistic features and anagraphic features
	
    model = Model(inputs=[deep_inputs, pos_tagging,other_features], outputs=output)

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    return model

# Reading the dataframe produced by the preprocessing steps. 

df = pd.read_pickle('data/pitt_full_interview_features.pickle')

# Conversion of the data from categorical labels to numerical labels that can be used by our algorithm
numeric_label = []
for string in df.label:
    if string == 'Dementia':
        numeric_label.append(1)
    if string == 'Control':
        numeric_label.append(0)

# Conversion of the 
def manual_features_conversion(manual_features):
    len_features = len(manual_features[0])

    final_list = []
    for element in manual_features:
        j_list = []
        for j in element:
            j_list.append(j)
        final_list.append(j_list)

    return np.array(final_list).reshape(len(manual_features),len_features,1)


def unpack_tup(tuple):
    return tuple[1]

# Computing the sequence of the POS tags for each interview. 
def compute_pos_tagged_sequence(input_list):
    print('Computing POS ')
    pos_tags_set = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT',
                     'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP',
                     'VBZ', 'WDT', 'WP', 'WP$', 'WRB']

    sentences_tag = []

    for item in input_list:
        tagged = nltk.pos_tag(item)
        tokenized_list = list(map(unpack_tup,tagged))
        sentences_tag.append(tokenized_list)


    word_to_id = {token: idx for idx, token in enumerate(set(pos_tags_set))}
    token_ids = [[word_to_id[token] if token != "''" else 0 for token in tokens_doc] for tokens_doc in sentences_tag]

    X = []
    for lst in token_ids:
        one_hot_encoded_list = np.zeros(len(word_to_id))
        for element in lst:
            one_hot_encoded_list[element] +=1
        X.append(one_hot_encoded_list)
    return X

# Starting the model training phase, we will run our experiments three times with three different random seeds. 
result_list = []
for seed in [4,10,95]:

	# train and test split 
    df_train, df_test, y_train, y_test = train_test_split(df, np.array(numeric_label), test_size=0.1, random_state=seed)
    df_train, df_validation, y_train, y_validation = train_test_split(df_train,y_train, test_size=0.1,random_state=seed)
	
	# preparation of the hand-crafted feature sequences 
    manual_feat_train = manual_features_conversion(df_train.features.values)
    manual_feat_test = manual_features_conversion(df_test.features.values)
    manual_feat_validation = manual_features_conversion(df_validation.features.values)

    len_features = manual_feat_train.shape[1]
	
	
    text_train = df_train.text
    text_validation = df_validation.text
    text_testing = df_test.text

	# tokenization of the interview 
    tokenizer = Tokenizer(num_words= vocabulary_size)
    tokenizer.fit_on_texts(text_train)
	
	# transformation of the tree interview set in sequences and POS tag computation 
    train_sequences = tokenizer.texts_to_sequences(text_train)
    train_sequences = pad_sequences(train_sequences, maxlen=sequence_len)
    train_tagged = compute_pos_tagged_sequence(text_train)

    validation_sequences = tokenizer.texts_to_sequences(text_validation)
    validation_sequences = pad_sequences(validation_sequences, maxlen=sequence_len)
    validation_tagged = compute_pos_tagged_sequence(text_validation)

    test_sequences = tokenizer.texts_to_sequences(text_testing)
    test_sequences = pad_sequences(test_sequences, maxlen=sequence_len)
    test_tagged = compute_pos_tagged_sequence(text_testing)

    #Word embeddings initialization
    embeddings_index = dict()
    f = open('glove.6B/glove.6B.'+str(EMBEDDING_SIZE)+'d.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    embedding_matrix = np.zeros((vocabulary_size, EMBEDDING_SIZE))
    for word, index in tokenizer.word_index.items():
        if index > vocabulary_size - 1:
            break
        else:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector

    #Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=1, mode='auto')
    tensor_borad = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
	
	#Optimizer we are using for training 
    optimizer = Adagrad(lr=0.001, epsilon=None, decay=0.0)

	# Model initialization
    model =  create_CNN_LSTM_POS_model_attention(vocabulary_size, sequence_len, embedding_matrix, EMBEDDING_SIZE,pos_tag_len,len_features)

    print("[LOG] Training the model...")

    #preparing the sequences for training and testing. 
    
    train_POS = np.array(train_tagged).reshape((np.array(train_tagged).shape[0],np.array(train_tagged).shape[1],1))
	
	test_POS = np.array(test_tagged).reshape((np.array(test_tagged).shape[0],np.array(test_tagged).shape[1],1))

    x_input = [train_sequences, train_POS, manual_feat_train]

    validation_POS = np.array(validation_tagged).reshape((np.array(validation_tagged).shape[0], np.array(validation_tagged).shape[1], 1))

    validation_dat = [validation_sequences, validation_POS, manual_feat_validation]
	
	# class weights passed to the .fit() keras method, this will force the model to pay 6 times more attention to the 0 class (it represent the healty patient)
	
    class_weight = {
        0: 6.,
        1: 1.,
    }
	
	# Actual model training 
    model.fit(x=x_input,y=y_train, validation_data=(validation_dat, y_validation),epochs=300,callbacks=[early_stopping,tensor_borad],verbose=0, class_weight=class_weight)


	# model prediction on the test set.  
    result = model.predict([test_sequences, test_POS, manual_feat_test])
    
    # the prediction of the algorithm is converted to binary prediction considering a classification threshold of 0.5
    # any floating point number outputed by our algorithm is converted to 0 if it is below 0.5 or to 1 if it is greater or equal than 0.5 
    
    y_score = binary_conversion(result,0.5)
    
    # evaluation metrics on the test set are computed 
    test_accuracy = accuracy_score(y_test,y_score)
    test_f1 = f1_score(y_test,y_score)
    confusion_matrix = confusion_matrix(y_test,y_score)
    precision = precision_score(y_test,y_score)
    recall = recall_score(y_test,y_score)
    
    print("Test accuracy: {}, Test F1 score: {}, with classification threshold 0.5".format(test_accuracy,test_f1))
    print(confusion_matrix.ravel())
    
    #AUC is computed 
    
    fpr, tpr, _ = roc_curve(y_test, result)
    roc_auc = auc(fpr, tpr)
    print("Precision: {}, Recall: {}, AUC: {}".format(precision,recall,roc_auc))
	
	# A result dictionary is produced for this run
    result_dictionary = {'Test Accuracy': test_accuracy, 'Test F1': test_f1, 'Precision': precision, 'Recall': recall,
                         'Confusion': confusion_matrix, 'AUC': roc_auc}
    result_list.append(result_dictionary)


print('Final results for the tests')

# the list of the results obtained for each run is printed. 
print(result_list)
