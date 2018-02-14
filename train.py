import tensorflow as tf
import numpy as np
import time

import pretrain
import utils
import RNN

DATA_DIR = './data/Ann.txt'
BATCH_SIZE = 100    # Sequences per batch
NUM_STEPS = 100    # Number of sequence steps per batch
LSTM_SIZE = 512    # Size of hidden layers in LSTMs
NUM_LAYERS = 2    # Number of LSTM layers
LEARNING_RATE = 0.001    # Learning rate
KEEP_PROB = 0.5    # Dropout keep probability

epochs = 20

# Print losses every N interations
print_every_n = 50

# Save every N interations
save_every_n = 200

encoded, int_to_vocab, vocab, _ = utils.load_data( DATA_DIR )
model = RNN.CharRNN( len( vocab ), batch_size = BATCH_SIZE, num_steps = NUM_STEPS,
                 lstm_size = LSTM_SIZE, num_layers = NUM_LAYERS, learning_rate = LEARNING_RATE)

saver = tf.train.Saver( max_to_keep = 100 )
with tf.Session() as sess:
    sess.run( tf.global_variables_initializer() )

    #Use the line below to load a checkpoint and resume training
    # saver.restore( sess, 'checkpoints/check.ckpt' )
    counter = 0
    for e in range( epochs ):
        # Train network
        new_state = sess.run( model.initial_state )
        loss = 0
        for x, y in pretrain.get_batches( encoded, BATCH_SIZE, NUM_STEPS ):
            counter += 1
            start = time.time()
            feed = {model.inputs : x,
                    model.targets : y,
                    model.keep_prob : KEEP_PROB,
                    model.initial_state : new_state}
            batch_loss, new_state, _ = sess.run( [model.loss,
                                                  model.final_state,
                                                  model.optimizer],
                                                  feed_dict = feed)
            if ( counter % print_every_n == 0 ):
                end = time.time()
                print( 'Epoch: {}/{}...'.format( e + 1, epochs ),
                       'Training Step: {}...'.format( counter ),
                       'Training loss: {:.4f}...'.format( batch_loss ),
                       '{:.4f} sec/batch'.format( ( end - start ) ) )

            if ( counter % save_every_n == 0 ):
                saver.save( sess, "checkpoints/i{}_1{}.ckpt".format( counter, LSTM_SIZE ) )

    saver.save( sess, "checkpoints/i{}_1{}.ckpt".format( counter, LSTM_SIZE ) )