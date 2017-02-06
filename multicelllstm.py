import tensorflow as tf
import math
import numpy as np
from tensorflow.python.ops.rnn_cell import RNNCell


class MultiCellLSTM(tf.nn.rnn_cell.RNNCell):


    def __init__(self, num_units,  num_cells, forget_bias=1.0, activation=tf.nn.tanh):
        self._num_units = num_units
	self._num_cells = num_cells
        self._forget_bias = forget_bias
        self._activation = activation


    @property
    def state_size(self):
        return tf.nn.rnn_cell.LSTMStateTuple(self._num_units*self._num_cells, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    #TODO: add forget bias
	
    def __call__(self, inputs, state, scope=None):

	batch_size = inputs.get_shape()[0]
	c, h = state	
	cell_size = [-1, self._num_cells, self._num_units]


        with tf.variable_scope(scope or type(self).__name__):
            
	            with tf.variable_scope("Gates"):  # Reset gate and update gate.
			            ifo = tf.nn.rnn_cell._linear([inputs, h], 3 * self._num_units, True)
				    ifo = tf.nn.sigmoid(ifo)
		                    i, f, o = tf.split(1, 3, ifo)
					                                    
		    with tf.variable_scope("Attention"):	
				    p = tf.nn.softmax(tf.nn.rnn_cell._linear([inputs, h],  self._num_cells, True))
	
            	    with tf.variable_scope("Candidate"):
				    cc = self._activation(tf.nn.rnn_cell._linear([inputs, h], self._num_units, True))
				   
				    #pp = tf.reshape(tf.tile(p, tf.pack([1, self._num_units])), cell_size1)
				    #pp = tf.transpose(pp, [0, 2, 1])	
				    #pp = tf.reshape(tf.tile(p, tf.pack([1, self._num_units])), cell_size)

				    # should be equatl to pp 		
				    pp0 = tf.reshape(p, [-1])
				    pp1 = tf.tile(pp0, tf.pack([self._num_units]))		   
				    pp2 = tf.reshape(pp1, [self._num_units, -1])
				    pp3 = tf.transpose(pp2, [1, 0])
				    pp =  tf.reshape(pp3, cell_size)	
				    #print("pp {}".format(pp.get_shape()))
				    
				    ff = tf.reshape(tf.tile(f, tf.pack([1, self._num_cells])), cell_size)
				    ii = tf.reshape(tf.tile(i, tf.pack([1, self._num_cells])), cell_size)
			 	    C_tilde = tf.reshape(tf.tile(cc, tf.pack([1, self._num_cells])), cell_size)
				    C_old = tf.reshape(c, cell_size) 

				    #print("ff {}".format(ff.get_shape()))

			            C_new = (C_old * ff   + C_tilde * ii) * pp 

				    mean_c = tf.reduce_sum(C_new, 1)
				    new_c = tf.reshape(C_new, [-1, self._num_cells * self._num_units])					


	  	    new_h = o * self._activation(mean_c)
	  	    new_state = tf.nn.rnn_cell.LSTMStateTuple(new_c, new_h)

        return new_h, new_state
