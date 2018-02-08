import tensorflow as tf
import tensorflow.contrib.slim as slim
from gen_dataset import *
from model import *
import config
import utils
import os 

def calc_validation_loss(sess, loss, accuracy, input_seq, ouput_seq, X_val, y_val):
    '''
    Calculate validation loss on the entire validation set
    '''
    val_accuracy, val_loss, val_batches = 0., 0., 0
    batch_size = min(config.val_batch_size, X_val.shape[0])
    for (inputs, targets) in utils.iterate_minibatches(X_val, y_val, batchsize=batch_size):

        batch_loss, batch_accuracy = sess.run([loss, accuracy], feed_dict={input_seq : inputs, ouput_seq : targets})
        val_batches += 1
        val_loss += batch_loss
        val_accuracy += batch_accuracy

    val_loss /= val_batches
    val_accuracy /= val_batches
    return val_loss, val_accuracy

def get_global_step_init(experiment_dir):
    '''
    Calculate proper global_step initialization
    '''
    ckpt = tf.train.get_checkpoint_state(experiment_dir)
    global_step_init = 0
    if ckpt and ckpt.model_checkpoint_path:
        global_step_init = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])

    return global_step_init

def main():

    experiment_dir = os.path.dirname(os.path.abspath(__file__))

    # generating the dataset
    X_train, X_val, y_train, y_val = gen_dataset(config.op, config.op_sym, config.training_size, config.digits)
    
    # define input placeholders 
    with tf.name_scope('inputs'):
        input_seq = tf.placeholder(tf.float32, [None, config.maxlen, config.numbers_alphabet_size + 2], name='input_seq') # tensor of shape <batch_size, seq_len, embedding size>
        expected_ouput_seq = tf.placeholder(tf.float32, [None, config.maxlen, config.numbers_alphabet_size + 1], name='ouput_seq') # tensor of shape <batch_size, seq_len, numbers_alphabet_size>
        
    # construct layers
    nn_output_probs = construct_layers(input_seq) 
    nn_ouput_probs_rsh = tf.reshape(nn_output_probs, [-1, config.numbers_alphabet_size + 1])
   
    expected_ouput_seq_rsh = tf.reshape(expected_ouput_seq, [-1, config.numbers_alphabet_size + 1])

 

    # define loss 
    with tf.name_scope('loss'):    
        loss = tf.losses.softmax_cross_entropy(onehot_labels=expected_ouput_seq_rsh, logits=nn_ouput_probs_rsh)
        #t
    loss_summary = tf.summary.scalar('loss', loss)

    # define accuracy
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(expected_ouput_seq_rsh, 1), tf.argmax(nn_ouput_probs_rsh, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accuracy_summary = tf.summary.scalar('accuracy', accuracy)

    # optimization
    optimizer = tf.train.AdamOptimizer(config.lr)
    global global_step
    global_step = tf.train.get_or_create_global_step()
    train_op = slim.learning.create_train_op(loss, optimizer, global_step=global_step)

    # config magic
    tf_config = tf.ConfigProto(log_device_placement = False)
    tf_config.gpu_options.allow_growth = True
    tf_config.gpu_options.allocator_type = 'BFC'
    
    try:
        with tf.train.MonitoredTrainingSession(checkpoint_dir=experiment_dir,
                                               save_checkpoint_secs=300,
                                               save_summaries_steps=200, config=tf_config) as sess:

            global_step_init = get_global_step_init(experiment_dir) #TODO
            if global_step_init != 0:
                print('Last run was aborted on global_step_init :{}'.format(global_step_init))
            else:
                print('Previous runs not found (You are begining this experiment).')


            for inputs, targets in utils.iterate_minibatches_global(X_train, y_train, batchsize=config.batch_size, start_it=global_step_init, iters=config.iters):

                train_loss = sess.run([train_op, loss, accuracy], feed_dict={input_seq : inputs, expected_ouput_seq : targets})
                gs = sess.run(global_step)

                if gs % config.print_loss_every == 0:
                    print("Step = {}; Train loss: {}".format(gs, train_loss))

                if gs % config.calc_val_loss_every == 0:
                    val_loss, val_accuracy = calc_validation_loss(sess, loss, accuracy, input_seq, expected_ouput_seq, X_val, y_val)
                    val_loss_summary = tf.Summary()
                    val_loss_summary.value.add(tag="loss/val_loss", simple_value=val_loss)
                    val_loss_summary.value.add(tag="accuracy/val_accuracy", simple_value=val_accuracy)
                    sess._hooks[1]._summary_writer.add_summary(val_loss_summary, gs)
                    print("=========== Step = {}; Val loss: {}; Val accuracy: {}".format(gs, val_loss, val_accuracy))

    except KeyboardInterrupt:
        print('Train process was interrupted')

    print('Done')

if __name__ == "__main__":
    main()

