

import numpy as np
import tensorflow as tf
from layers import *
from tensorflow.contrib.layers import fully_connected

def construct_layers(input_seq):
    '''
    input_seq : tensor of shape <batch_size, seq_len, embedding size>

    '''

    with tf.name_scope('s_tensor'):
        nn = tf.expand_dims(input_seq, 1) # tensor of shape <batch_size, 1, seq_len, embedding size>
        paddings = tf.constant([[0, 0], [0, config.maxlen - 1], [0, 0], [0, 0]])
        nn = tf.pad(nn, paddings, 'CONSTANT') # tensor of shape <batch_size, maxlen, seq_len, embedding size>
        
    with tf.name_scope('CGRU'):
        for i in range(config.maxlen * config.CGRU_apply_times):
            nn = conv_gru(nn, config.kw, config.kh, config.nmaps, config.cutoff, 'CGRU_apply_step_{}'.format(i))
        
    output = nn[:, 0, :, :] # tensor of shape <batch_size, time, embedding size>

    output_shape = output.get_shape().as_list()
    output = tf.reshape(output, [-1, output_shape[-1]])
    probs = fully_connected(output, config.numbers_alphabet_size + 1, activation_fn=tf.nn.softmax)
    print('Output shape: {}'.format(probs.get_shape().as_list()))
    return probs