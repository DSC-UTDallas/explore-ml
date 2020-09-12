"""
output.py - module to store BiLSTM-CRF model
"""

import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, TimeDistributed, Dropout, Bidirectional, Dense
from tensorflow.keras.initializers import Constant
from .loss import CRF

def embedding_map(glove_path = 'glove.6B.200d.txt'):
    """
    embedding_map - function to load weights of the pretrained Glove embedding file
    Parameters:
        glove_path              I/P     path to the pretrained Glove embedding file
        embedding_index		O/p	weights of the pretrained Glove embedding file
    """
    embeddings_index = {}
    with open(glove_path) as f:
        for line in f:
            word, coefs = line.split(maxsplit = 1)
            coefs = np.fromstring(coefs, 'f', sep = ' ')
            embeddings_index[word] = coefs
    print('Found %s word vectors.' %len(embeddings_index))
    return embeddings_index

def embedding_layer(word2dix, input_dim, output_dim, input_length, mask_zero):
    embedding_index = embedding_map()

    embedding_matrix = np.zeros((input_dim, output_dim))
    for word, i in word2dix.items():
      embedding_vector = embdding_index.get(word)
      if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

    return Embedding(input_dim = input_dim, output_dim = output_dim, input_length = input_length, trainable = False, embeddings_initializer = Constant(embedding_matrix))

def bilstm_crf(word2dix, maxlen, n_tags, embedding_dim, n_words, mask_zero, training = True):
    """
    bilstm_crf - module to build BiLSTM-CRF model
    Inputs:
        - input_shape : tuple
            Tensor shape of inputs, excluding batch size
    Outputs:
        - output : tensorflow.keras.outputs.output
            BiLSTM-CRF output
    """
    input = Input(shape = (maxlen,))
    # Embedding layer
    embeddings = embedding_layer(input_dim = n_words + 1, output_dim = embedding_dim, input_length = maxlen, mask_zero = mask_zero)
    output = embeddings(input)

    # BiLSTM layer
    output = Bidirectional(LSTM(units = 50, return_sequences = True, recurrent_dropout = 0.1))(output)

    # Dense layer
    output = TimeDistributed(Dense(n_tags, activation = 'relu'))(output)

    output = CRF(n_tags, name = 'crf_layer')(output)
    return Model(input, output)
