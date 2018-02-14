import tensorflow as tf
import numpy as np
import pretrain

class CharRNN:
    def __init__( self, num_classes, batch_size = 64, num_steps = 50, lstm_size = 128,
                  num_layers = 2, learning_rate = 0.001, grad_clip = 5, sampling = False ):
        # When we're using this network for smpling layer, we'll be passing in one
        # character at a time, so providing an option for that
        if sampling == True:
            batch_size, num_steps = 1, 1
        else:
            batch_size, num_steps = batch_size, num_steps

        tf.reset_default_graph()

        # Build the input placeholder tensors
        self.inputs, self.targets, self.keep_prob = pretrain.build_inputs( batch_size, num_steps )

        # Build the LSTM cell
        cell, self.initial_state = pretrain.build_lstm( lstm_size, num_layers, batch_size, self.keep_prob )

        ### Run the data through the RNN layers
        # First, one-hot encode the input tokers
        x_one_hot = tf.one_hot( self.inputs, num_classes )

        # Run each sequence step through the RNN and collect the output
        outputs, state = tf.nn.dynamic_rnn( cell, x_one_hot, initial_state = self.initial_state )
        self.final_state = state

        # Get softmax predictions and logits
        self.prediction, self.logits = pretrain.build_output( outputs, lstm_size, num_classes )

        # Loss and optimizer ( with gradient clipping )
        self.loss = pretrain.build_loss( self.logits, self.targets, lstm_size, num_classes )
        self.optimizer = pretrain.build_optmizer( self.loss, learning_rate, grad_clip )