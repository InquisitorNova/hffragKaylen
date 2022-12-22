"""
The module file which lays out the loss and convolutional 
recurrent neural network architecture for the regression of b jets.
"""
#Set up the coding standard and point the interpreter to the correct
#Virtual environment

#!/usr/bin/env python3
# coding: utf-8

#Important relevant modules
import numpy as np
import awkward as ak
import tensorflow as tf
import keras
from keras import layers
from keras import regularizers

#Initial Parameters 
MASKVAL = -999

def LogNormal_Loss_Function(true,mean_convariance_matrix):
    
    """A custom loss function designed to force the neural network 
    to return a prediction and associated uncertainty for target features"""

    #Identify the number of target features
    n_targets = np.shape(true)[1]

    #Allocate the first n outputs of the dense layer to represent the mean
    means = mean_convariance_matrix[:, :n_targets]

    #Allocate the second n outputs of the dense layer to represent the variances
    logvariances = mean_convariance_matrix[:, n_targets: 2* n_targets]

    #Allocate the last n outputs of th4e dense layer to represent the covariances
    logcovariances = mean_convariance_matrix[:, 2*n_targets:]


    #Calculate the logNormal loss
    sum_loss = 0
    for target in range(n_targets):
        sum_loss += (1/2)*keras.backend.log(2*np.pi) + logvariances[:,target] + ((true[:,target] - means[:,target])**2)/(2*keras.backend.exp(logvariances[:,target])**2)
    
    return sum_loss

def Root_Mean_Square_Metric(true, mean_convariance_matrix):

    """
    A custom metric used to discern the accuracy of the model without influencing
    how the models weights and biases are adjusted
    """
    #Determine the number of targets
    n_targets = np.shape(true)[1]

    #Select the predicted values of the targets
    means = mean_convariance_matrix[:, :n_targets]

    #Determine the root mean square of the values
    diff = tf.math.subtract(true,means)
    square = tf.square(diff)
    mean_square_error = tf.math.reduce_sum(square)
    #Return the accuracy
    return tf.math.sqrt(mean_square_error).numpy()

class LNGRU(layers.Layer):

    """A custom layer that normalisers the outputs per GRU layer"""

    def __init__(self, units, dropout, activation = "tanh", **kwargs):
        #Creates a constructor which inherits from the keras layer model
        super().__init__(**kwargs)
        self.state_size = units # Sets the size of the state vector used to provide context
        self.output_size = units # Sets the size of the output from the cell 
        self.gru_cell = layers.GRUCell(units,activation, dropout = dropout) # Creates the GRU cell

        self.layer_norm = layers.LayerNormalization() #Creates the normalisation layer
        self.activation = keras.activations.get(activation) # Creates the activation function used

    def call(self, inputs, states): #Activates upon being called
        outputs, new_states = self.gru_cell(inputs,states) # Runs the gru cell returns the altered state and outputs
        norm_outputs = self.activation(self.layer_norm(outputs)) #Normalises the results and then passes to activation function
        return norm_outputs, new_states



def ConvolutionRecurrentNeuralNetwork(track_layers,n_targets,optimizer,dropout = 0.0, MASKVAL = -999):

    """This function defines the neural network architecture for a convolution recurrent neural network. 
    It combines a 1D convolution layer with GRU layers in order to produce a neural network capable
    from learning from non-linear data with a undefined length. While less efficient compared to a DeepSetNeural Network
    when it comes to convergence rate, the intention of this neural network is to act as a benchmark."""

    inputs = layers.Input(shape = (None,track_layers[0])) # Creates an input layer of undefined length.
    outputs = inputs #Realiasing the input layer into an output layer
    outputs = layers.Masking(mask_value = MASKVAL)(outputs) # Tells the network to ignore the -999
    
    outputs = layers.Conv1D(filters = 10, kernel_size = 4, strides = 2, padding = "valid", activation = "elu" )(outputs)

    for nodes in track_layers[1:-1]:
        outputs = layers.RNN(LNGRU(nodes,dropout), return_sequences = True)(outputs)
        outputs = layers.BatchNormalization()(outputs)

    outputs = layers.RNN(LNGRU(nodes,dropout))(outputs)
    outputs = layers.Dense(n_targets+n_targets*(n_targets+1)//2)(outputs)

    Model = keras.Model(inputs = inputs, outputs = outputs)

    Model.compile(
        optimizer = optimizer, # Optimizer used to train the model
        metrics = [Root_Mean_Square_Metric], # Metric used to assess the true performance of the model
        loss = LogNormal_Loss_Function, # Loss function
        run_eagerly = True #Allows Numpy to run
        )

    return Model

