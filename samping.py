import tensorflow as tf
import numpy as np

import RNN
import utils

DATA_DIR = './data/Ann.txt'
LSTM_SIZE = 512

encoded, int_to_vocab, vocab, vocab_to_int = utils.load_data( DATA_DIR )

def pick_top_n( preds, vocab_size, top_n = 5 ):
    p = np.squeeze( preds )
    p[np.argsort( p )[: -top_n]] = 0
    p = p / np.sum( p )
    c = np.random.choice( vocab_size, 1, p = p)[0]
    return c

def sample( checkpoint, n_samples, lstm_size, vocab_size, prime = "The " ):
    samples = [c for c in prime]
    model = RNN.CharRNN( len( vocab ), lstm_size = lstm_size, sampling = True )
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore( sess, checkpoint )
        new_state = sess.run( model.initial_state )
        for c in prime:
            x = np.zeros( ( 1, 1 ) )
            x[0, 0] = vocab_to_int[c]
            feed = { model.inputs : x,
                     model.keep_prob : 1.,
                     model.initial_state : new_state}
            preds, new_state = sess.run( [model.prediction, model.final_state],
                                         feed_dict = feed )
        c = pick_top_n( preds, len( vocab ) )
        samples.append( int_to_vocab[c] )

        for i in range( n_samples ):
            x[0, 0] = c
            feed = { model.inputs : x,
                     model.keep_prob : 1.,
                     model.initial_state : new_state}
            preds, new_state = sess.run( [model.prediction,
                                          model.final_state],
                                          feed_dict = feed )
            c = pick_top_n( preds, len( vocab ) )
            samples.append( int_to_vocab[c] )

    return ''.join( samples )

if __name__ == "__main__":
    encoded, int_to_vocab, vocab, vocab_to_int = utils.load_data(DATA_DIR)
    checkpoint = tf.train.latest_checkpoint( 'checkpoints' )
    samp = sample( checkpoint, 2000, LSTM_SIZE, len( vocab ), prime = "Far" )
    print( samp )

    checkpoint = 'checkpoints/i200_1512.ckpt'
    samp = sample(checkpoint, 1000, LSTM_SIZE, len(vocab), prime = "Far")
    print(samp)

    checkpoint = 'checkpoints/i600_1512.ckpt'
    samp = sample(checkpoint, 1000, LSTM_SIZE, len(vocab), prime = "Far")
    print(samp)

    checkpoint = 'checkpoints/i1200_1512.ckpt'
    samp = sample( checkpoint, 1000, LSTM_SIZE, len(vocab), prime = "Far" )
    print(samp)