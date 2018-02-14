import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn

def get_batches( arr, batch_size, n_steps ):
    """
    Creat a generator that returns batches of size batch_size x n_steps from arr.

    Arguments
    _________
    arr: Array you want to make batches from
    batch_size: Batch size, the nunber of sequences per batch
    n_steps: Number of sequence step per batch
    """
    # Get the number of characters per batch and number of batches we can make
    chars_per_batch = batch_size * n_steps
    n_batches = len( arr ) // chars_per_batch    #除法取整

    # Keep only enough charcters to meke full batches
    arr = arr[: n_batches * chars_per_batch]

    # Reshape into batch_size rows
    arr = arr.reshape( ( batch_size, -1 ) )

    for n in range( 0, arr.shape[1], n_steps ):
        # The features
        x = arr[:, n : n_steps + n]
        #The tragets, shifted by one
        y_temp = arr[:, n + 1 : n + 1 + n_steps]
        '''
        For the very last batch, y will be one character shorter at the end of
        the sequences which breaks things, To get around this, I'll make an array
        of the approirate size first, of all zeros, then add the targets.
        This will introduce a small artifact in the last batch, but it won't matter
        '''
        y = np.zeros( x.shape, dtype=x.dtype )
        y[:,: y_temp.shape[1]] = y_temp

        yield x, y

def build_inputs( batch_size, num_steps ):
    '''
    Define placeholder for inputs, targets, and output

    Arguments
    ------------
    batch_size: Batch size, number of sequences per batch
    num_steps: Number of sequence steps in a batch
    '''
    # Declare placeholder we'll feed into the graph
    inputs = tf.placeholder( tf.int32, [batch_size, num_steps], name = 'input' )
    targets = tf.placeholder( tf.int32, [batch_size, num_steps], name = 'targets' )

    # Keep probability placeholder for drop out layers
    keep_prob = tf.placeholder( tf.float32, name = 'keep_prob' )

    return inputs, targets, keep_prob

def build_lstm( lstm_size, num_layers, batch_size, keep_prob ):
    '''
    Build LSTM

    Arguments
    ----------
    lstm_size: Size of the hidder layers in the LSTM cell
    num_layers: Number of LSTM layers
    batch_size: Batch size
    keep_probs: Scalar tensor ( tf.placehoder ) for the
    '''

    ### Build the LSTM Cell

    def build_cell( lstm_size, keep_prob ):
        # Use a basic LSTM cell
        lstm = rnn.BasicLSTMCell( lstm_size )

        # Add dropout to cell
        drop = rnn.DropoutWrapper( lstm, output_keep_prob = keep_prob )
        return drop

    # Stack up multiple LSTM layers, for deep learning
    cell = rnn.MultiRNNCell( [build_cell( lstm_size, keep_prob ) for _ in range( num_layers )] )
    initial_state = cell.zero_state( batch_size, tf.float32 )

    return cell, initial_state

def build_output( lstm_output, in_size, out_size ):
    '''
    Build a softmax layer, return the softmax output and logits

    Arguments
    ---------

    lstm_output: Input tensor
    in_size: Size of the input tensor, for example, size of the LSTM cells
    out_size: Size of this softmax layer
    '''

    # Reshape output so it's a bunch of rows, one row for each for each sequence.
    # That is, the shape should be batch_size * num_steps rows by lstm_size columns
    seq_output = tf.concat( lstm_output, axis = 1 )    # 把数组按维度链接起来，axis维度组合，0一维，1二维
    x = tf.reshape( seq_output, [-1, in_size] )

    # Connect the RNN outputs to a softmax layer
    with tf.variable_scope( 'softmax' ):
        softmax_w = tf.Variable( tf.truncated_normal( ( in_size, out_size ), stddev = 0.1 ) )
        softmax_b = tf.Variable( tf.zeros( out_size ) )

    # Since output is a bunch of rows of RNN cell outputs, logits will be a bunch
    # of rows of logit outputs, one for each step and sequence
    logits = tf.matmul( x, softmax_w ) + softmax_b

    #Use softmax to get the probabilities for pridicted characters
    out = tf.nn.softmax( logits, name = 'predictions' )

    return out, logits

def build_loss( logits, targets, lstm_size, num_classes ):
    '''
    Caculate the loss from the logits and the targets.

    Arguments
    ---------
    logits: Logits from final fully connected layer
    targets: Targets for supervised learning
    lstm_size: Number of LSTM hidden units
    num_classes: Number of classes in targets
    '''

    # One-hot encode targets and reshape to much logits, one row per batch_size per step
    y_one_hot = tf.one_hot( targets, num_classes )
    y_reshaped = tf.reshape( y_one_hot, logits.get_shape() )

    #Softmax cross entropy loss
    loss = tf.nn.softmax_cross_entropy_with_logits( logits = logits, labels = y_reshaped )
    loss = tf.reduce_mean( loss )
    return loss

def build_optmizer( loss, learning_rate, grad_clip ):
    '''
    Build optmizer for training, using gradient clipping

    Arguments:
    ---------
    loss: Network loss
    learning_rate: Learning rate for optimizer
    '''

    #Optimizer for training, using gradient clipping to control exploding gradients
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm( tf.gradients( loss, tvars ), grad_clip )
    train_op = tf.train.AdamOptimizer( learning_rate )
    optimizer = train_op.apply_gradients( zip( grads, tvars ) )

    return optimizer