import tensorflow as tf
from gen_dataset import *
from model import *
import config


def main():

    # generating the dataset
    dataset = gen_dataset(config.op, config.op_sym, config.training_size, config.digits)
    
    # define input placeholders 
    with tf.name_scope('inputs'):
        input_seq = tf.placeholder(tf.float32, [None, config.maxlen, config.numbers_alphabet_size + 2], name='input_seq') # tensor of shape <batch_size, seq_len, embedding size>
        ouput_seq = tf.placeholder(tf.float32, [None, config.maxlen, config.numbers_alphabet_size + 1], name='ouput_seq') # tensor of shape <batch_size, seq_len, numbers_alphabet_size>
        
    # construct layers
    output_probs = construct_layers() 
    ouput_seq_rsh = tf.reshape(output, [-1, config.numbers_alphabet_size + 1])
   
   # define loss 
    with tf.name_scope('loss'):    
        loss = tf.losses.softmax_cross_entropy(onehot_labels=ouput_seq_rsh, logits=output_probs)
    loss_summary = tf.summary.scalar('loss', loss)

   # define accuracy
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(ouput_seq, 1), tf.argmax(ouput_seq_rsh, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accuracy_summary = tf.summary.scalar('accuracy', accuracy)


if __name__ == "__main__":
    main()