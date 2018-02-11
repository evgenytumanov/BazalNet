

import numpy as np
import tensorflow as tf
from layers import *
from tensorflow.contrib.layers import fully_connected

def construct_layers(input_seq):
    '''
    input_seq : tensor of shape <batch_size, seq_len, embedding size>

    '''

    with tf.name_scope('s_tensor'):
        input_shape = input_seq.get_shape().as_list()

        nn = tf.reshape(input_seq, [-1, input_shape[-1]])
        nn = fully_connected(nn, config.nmaps)
        print('Embed: {}'.format(nn.get_shape().as_list()))

        nn = tf.reshape(nn, [-1, input_shape[1], config.nmaps])
        print('Embed seq: {}'.format(nn.get_shape().as_list()))

        nn = tf.expand_dims(nn, 1) # tensor of shape <batch_size, 1, seq_len, nmaps>
        
        print('paddings input: {}'.format(nn.get_shape().as_list()))
        paddings = tf.constant([[0, 0], [0, config.mental_width - 1], [0, 0], [0, 0]])
        nn = tf.pad(nn, paddings, 'CONSTANT') # tensor of shape <batch_size, maxlen, seq_len, nmaps>
       
    print('CGRU input: {}'.format(nn.get_shape().as_list()))
    with tf.name_scope('CGRU'):
        for i in range(config.maxlen * config.CGRU_apply_times):
            nn = conv_gru(nn, config.kw, config.kh, config.nmaps, config.cutoff, 'CGRU_apply_step_{}'.format(i))
    print('conv_gru output: {}'.format(nn.get_shape().as_list()))
    output = nn[:, 0, :, :] # tensor of shape <batch_size, time, nmaps>

    output_shape = output.get_shape().as_list()
    output = tf.reshape(output, [-1, output_shape[-1]])
    probs = fully_connected(output, config.numbers_alphabet_size + 1, activation_fn=tf.nn.softmax)
    print('Output shape: {}'.format(probs.get_shape().as_list()))
    return probs