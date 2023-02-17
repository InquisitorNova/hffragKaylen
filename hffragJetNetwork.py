"""
*Filename: hffragJetNetwork
*Description: This python file lays out the neural network architecture for a
* Jet neural network used to predict the mean and variances of
* the targets. The python file also includes the definition for the loss the 
* neural network uses during training and its callbacks. This neural network 
* is simply a feedforward network used to benchmark 
* the performance of the other DeepSet network architectures.
Date: 16/02/2023
Author: Kaylen Smith Darnbrook
"""

# Import relevant modules
import awkward as ak
import numpy as np
import tensorflow as tf
import keras.layers as layers
from keras import regularizers
import keras.backend as K
import keras
from Sum import Sum
import matplotlib.pyplot as plt
from hffrag import fixedbinning
from hffrag import binneddensity

#Create global variables to store the predictions per epoch
Predicted_Bhad_px = np.array([])
Predicted_Bhad_px_uncertainties = np.array([])

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

# A callback returns a scatterplot of the predicted pxs and associated uncertainties per epoch
class PredictOnEpoch(tf.keras.callbacks.Callback):
    def __init__(self, model, x_test):
        self.model = model
        self.x_test = x_test
    
    def on_epoch_end(self, epoch, logs = {}):
        Means, MeansCovs = self.model.predict(self.x_test)
        px_pred = MeansCovs[:,0]
        px_uncertainity = MeansCovs[:,3]

        global Predicted_Bhad_px
        global Predicted_Bhad_px_uncertainties
        
        if epoch > 1:
            Predicted_Bhad_px = np.concatenate((Predicted_Bhad_px,px_pred))
            Predicted_Bhad_px_uncertainties = np.concatenate((Predicted_Bhad_px_uncertainties, px_uncertainity))
        else:
            Predicted_Bhad_px = px_pred
            Predicted_Bhad_px_uncertainties = px_uncertainity

#Calls the bhadsNet to reflect that it is a jet network regressing the bhads targets using only jet information. This neural network is a feed forward network
#used to regress the bhadron features from the jet features. It acts as the benchmark for other DeepSet networks to improve upon.
def bhadsNet(n_features,n_targets,NumDropout,jet_layers,Dropout_rate, MASKVAL = -999):
    
    #Load in the jet features. 
    jets = layers.Input(shape = (n_features))
    Num_Meancovs = n_targets+n_targets*(n_targets+1)//2
    
    #Mask out the padding applied to the jet information.
    output_bjets = jets
    output_bjets = layers.BatchNormalization()(output_bjets)
    output_bjets = layers.Masking(mask_value=MASKVAL)(output_bjets)

    outputs_Target_jets = output_bjets
    outputs_Uncertainties_jets = output_bjets

    #Apply a series of dense layers to extract information from the bhadron features and regress the bhadron features.
    #Two branchs of dense layers are used, one for regressing the targets and another for regressing the uncertainties.
    counter = NumDropout
    for nodes in jet_layers: #Repeat of the track neural network without the need for the timedistributed layers
        outputs_Target_jets = layers.Dense(nodes, activation='gelu', kernel_initializer= "he_normal", kernel_regularizer = regularizers.l1_l2(0.001))(outputs_Target_jets)
        
        if  counter > 0:
            outputs_Target_jets = layers.Dropout(Dropout_rate)(outputs_Target_jets)
        else:
            counter -= 1
        
        outputs_Target_jets= layers.BatchNormalization()(outputs_Target_jets)
    outputs_Target_jets = layers.Dense(n_targets, activation='gelu', kernel_initializer= "he_normal", kernel_regularizer = regularizers.l1_l2(0.001))(outputs_Target_jets)

    counter = NumDropout
    for nodes in jet_layers: #Repeat of the track neural network without the need for the timedistributed layers
        outputs_Uncertainties_jets = layers.Dense(nodes, activation='gelu', kernel_initializer= "he_normal", kernel_regularizer = regularizers.l1_l2(0.001))(outputs_Uncertainties_jets)
        
        if  counter > 0:
            outputs_Uncertainties_jets = layers.Dropout(Dropout_rate)(outputs_Uncertainties_jets)
        else:
            counter -= 1
        
        outputs_Uncertainties_jets= layers.BatchNormalization()(outputs_Uncertainties_jets)
    outputs_Uncertainties_jets = layers.Dense(Num_Meancovs - n_targets, activation='gelu', kernel_initializer= "he_normal", kernel_regularizer = regularizers.l1_l2(0.001))(outputs_Uncertainties_jets)
     
    #Combine the outputs of both branches to produce the final outputs.
    jet_out = layers.concatenate([outputs_Target_jets,outputs_Uncertainties_jets])

    jet_out = layers.Dense(Num_Meancovs, name = "Jet_Out")(jet_out)
    outputs_Target_jets = layers.Dense(n_targets, name = "Target_Values")(outputs_Target_jets)
    
    Model = keras.Model(inputs = jets, outputs = [outputs_Target_jets, jet_out])

    return Model




