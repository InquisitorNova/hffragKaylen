#!/usr/bin/env python
# coding: utf-8
"""
*Filename: HyperTunerBigGun
*Description: In this jupyter notebook, bayesian optimisation is used to explore 
*the hyperparameter space of the base DeepSets neural network architecture. These results
*from these tuning are used as the hyperparameters for future experiments with the 
*deep sets neural network architecture.
Date: 16/02/2023
Author: Kaylen Smith Darnbrook
"""
# In[35]:

#Import the relevant modules
import awkward as ak
import os
import keras
import keras.layers as layers
import numpy as np
from Sum import Sum
from keras import callbacks
from keras import regularizers
import uproot
import nbimporter
import import_ipynb
import matplotlib.pyplot as plt
from hffrag import fixedbinning
from hffrag import binneddensity
import DeepSetNeuralNetArchitecture as DSNNA
from numpy.lib.recfunctions import structured_to_unstructured
from DeepSetNeuralNetArchitecture import LogNormal_Loss_Function
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.preprocessing import StandardScaler
import pandas as pd
import keras_tuner as kt


# In[2]:


# The data is being stored in a tree datastructure.
# We access the charm root using this command
tree = uproot.open("hffrag.root:CharmAnalysis")


# In[3]:


# Initial parameters
MASKVAL = -999 # This value is introduced to ensure arrays are regular (Of the same size). They will be masked later by the network
MAXTRACKS = 32 # This value is the maximum number of tracks allowed per event
BATCHSIZE = 300 # This is the batch size of the mini batches used during training
EPOCHS = 100 # This is the default number of epochs for which the neural network will train providing that early stopping does not occur
MAXEVENTS = 1e20 #This is the maximum number of events that will the program will accept
LR = 1e-3 #This is the default learning rate


# In[5]:


# Select the features we wish to study
track_features = ["AnalysisTracks_pt", "AnalysisTracks_eta", "AnalysisTracks_phi", "AnalysisTracks_z0sinTheta",
                  "AnalysisTracks_d0sig", "AnalysisTracks_d0", "AnalysisTracks_d0sigPV", "AnalysisTracks_d0PV"]
jet_features = ["AnalysisAntiKt4TruthJets_pt", "AnalysisAntiKt4TruthJets_eta", "AnalysisAntiKt4TruthJets_phi",
                "AnalysisAntiKt4TruthJets_ghostB_pt", "AnalysisAntiKt4TruthJets_ghostB_eta","AnalysisAntiKt4TruthJets_ghostB_phi"]


# In[6]:


# Read in the data from the root file
features = tree.arrays(jet_features+track_features, entry_stop=MAXEVENTS)


# In[7]:


# Select the events of interest
events = features[ak.sum(
    features["AnalysisAntiKt4TruthJets_pt"] > 25000, axis=1) > 0]


# In[8]:


# Displays the number of jets being trained on
jets = events[jet_features][:, 0]
print("The number of jets to train on is: ", len(jets))
print("The number of track features is: ",len(track_features))


# In[9]:


# Select tracks from the events
tracks = events[track_features]

# Match the tracks to the jets
matchedtracks = tracks[DSNNA.Match_Tracks(jets, tracks)]

# Pad and Flatten the data
matchedtracks = DSNNA.flatten(matchedtracks, MAXTRACKS)


# In[10]:


# Identify the the bottom jets and their associated tracks
bjets = ak.sum(jets["AnalysisAntiKt4TruthJets_ghostB_pt"] > 5000, axis=1) > 0
jets = jets[bjets]

# Obtain the pt, eta and phi of each b hadron jet
bhads_pt = jets["AnalysisAntiKt4TruthJets_ghostB_pt"][:, 0].to_numpy()
bhads_eta = jets["AnalysisAntiKt4TruthJets_ghostB_eta"][:,0].to_numpy()
bhads_phi = jets["AnalysisAntiKt4TruthJets_ghostB_phi"][:,0].to_numpy()

bhads = np.stack([bhads_pt,bhads_eta,bhads_phi],axis = -1) #Combine the momentum, eta and phi for each jet into one array

print("There are {} outputs".format(np.shape(bhads)[1])) # Display the number of target features the neural network will predict
matchedtracks = matchedtracks[bjets]
print("There are {} inputs".format(np.shape(matchedtracks)[1])) # Display the number of target features the neural network will use in it's predictions


# In[11]:


# Transform the jet and tracks to unstructed data.
jets = structured_to_unstructured(jets[jet_features[:-3]])
matchedtracks = structured_to_unstructured(matchedtracks)


# In[12]:


# Convert the coordinates of the b jets and tracks to cartesian coordinates
tracks_p = DSNNA.pt_eta_phi_2_px_py_pz_tracks(matchedtracks.to_numpy())
bhads = DSNNA.pt_eta_phi_2_px_py_pz_jets(bhads)

#Combine the momenta of the tracks with the rest of the track features to form the track dataset
tracks = np.concatenate([tracks_p,matchedtracks[:,:,3:].to_numpy()],axis = 2)


# In[13]:

#The tracks are standardized using scikit learn standard scaler.
Scaler = StandardScaler()
Num_events,Num_tracks,Num_features = np.shape(tracks)
tracks = np.reshape(tracks, newshape=(-1,Num_features))
tracks = Scaler.fit_transform(tracks)
tracks = np.reshape(tracks, newshape= (Num_events,Num_tracks,Num_features))


# In[14]:


# Split the data into training and validation sets.
X_train, X_valid, y_train, y_valid = train_test_split(
    tracks, bhads, train_size=0.8, random_state=42)


# In[15]:

#Single events cases
X_train_event, y_train_event = np.array([X_train[0]]), np.array([y_train[0]])
X_valid_event, y_valid_event = np.array([X_valid[0]]), np.array([y_valid[0]])
print(np.shape(X_train),np.shape(y_train))
print(np.shape(X_train_event),np.shape(y_train_event))


# In[16]:


#Check for the of the training and validation sets
print(np.shape(X_train), np.shape(X_valid))
print(np.shape(y_train), np.shape(y_valid))


# In[47]:


def DeepSetNeuralNetwork(hp):
    """
    This function lays out the Deep Set Neural Architecture
    - A neural network is applied first to the tracks to extract information from the tracks.
    - This information produces an ensemble space which, the outputs of which are then summed to produce
        the inputs for the next layer
    - A neural network is then applied to the jet data obtained from the tracks. 
        To perform current univariate regression.
    """
    MASKVAL = -999
    # Create the ranges of hyperparameters to explore
    # Create the ranges of hyperparameters to explore
    dropout_rate = hp.Float('dropout_rate', 0.0001,0.1)
    track_layer_Neurons = hp.Choice('track_layers_Num_Neurons',[16,32,64,128,256,512])
    jet_layer_Neurons = hp.Choice('jet_layers_Number_Neurons',[16,32,64,128,256,512])
    activation_func = hp.Choice('act_func',["relu","elu","selu", "gelu"])
    Num_tracks_layer = hp.Choice('Num_track_layers_Neurons',[1,2,3,4,5,6,7,8,9,10])
    Num_jets_layer = hp.Choice('Num_jet_layers_Neurons',[1,2,3,4,5,6,7,8,9,10])
    Learning_rate = hp.Float('learning_rate',1e-5,1e-2)
    Initializer_value = hp.Choice('Initalizers', ["he_normal", "lecun_normal","glorot_uniform"])
    regularizer_value = hp.Choice("regularizer", [1e-6,1e-5,1e-4,1e-3])
    dropout_frequency = hp.Choice("DropoutFrequency",[1,2,3,4,5,6,7,8,9,10])
    beta_1 = hp.Float("beta_1", 0.9,0.999)
    beta_2 = hp.Float("beta_2", 0.9,0.999)
    weight_decay = hp.Choice("Weight_Decay", [0.1,0.2,0.4,0.8,1.0])

    #Create the track and jet layers
    track_layers = [len(track_features)]+[track_layer_Neurons for x in range(Num_tracks_layer)]
    jet_layers  = [jet_layer_Neurons for x in range(Num_jets_layer)]

    #Set the number of targets being explored
    n_targets = 3

    inputs = layers.Input(shape=(None, track_layers[0])) # Creates a layer for each input
    outputs = inputs  # Creates another layer to pass the inputs onto the ouputs
    outputs = layers.Masking(mask_value=MASKVAL)(outputs) # Masks the MASKVAl values

    counter = 0
    for nodes in track_layers[:-1]:
        #The first neural network is a series of dense layers and is applied to each track using the time distributed layer
        outputs = layers.TimeDistributed( 
            layers.Dense(nodes, activation=activation_func, kernel_initializer= Initializer_value, kernel_regularizer = regularizers.l1_l2(regularizer_value)))(outputs) # We use relu and the corresponding he_normal for the activation function and bias initializer
        if counter % dropout_frequency == 0: # Every two layers apply a dropout
            outputs = layers.Dropout(dropout_rate)(outputs)
        else:
            counter += 1
    
        outputs = layers.BatchNormalization()(outputs) # Apply a batch norm to improve performance by preventing feature bias and overfitting

    outputs = layers.TimeDistributed(layers.Dense( 
        track_layers[-1], activation="softmax", kernel_initializer= Initializer_value))(outputs) # Apply softmax to ouput the results of the track neural network as probabilities
    outputs = Sum()(outputs) # Sum the outputs to make use of permutation invariance

    
    counter = 0
    for nodes in jet_layers: #Repeat of the track neural network without the need for the timedistributed layers
        outputs = layers.Dense(nodes, activation=activation_func, kernel_initializer= Initializer_value, kernel_regularizer = regularizers.l1_l2(regularizer_value))(outputs)
        
        if counter % dropout_frequency == 0:
            outputs = layers.Dropout(dropout_rate)(outputs)
        else:
            counter += 1
        
        outputs = layers.BatchNormalization()(outputs)
    

    outputs = layers.Dense(3)(outputs) # The output will have a number of neurons needed to form the mean covariance function of the loss func

    Model = keras.Model(inputs=inputs, outputs=outputs) #Create a keras model

    # Specify the neural network's optimizer and loss function
    Model.compile(
    optimizer=keras.optimizers.Nadam(learning_rate=Learning_rate, beta_1 = beta_1, beta_2 = beta_2, decay = weight_decay), # Optimizer used to train model
    #metrics = [Normal_Accuracy_Metric,Root_Mean_Square_Metric], # Metric used to assess true performance of model
    loss= tf.keras.losses.MeanSquaredError(),#Loss function
    #jit_compile = True
    )
    #run_eagerly = True #Allows Numpy to run
    
    return Model


# In[48]:

# Defines a keras tuner which uses bayesian optimisation to explore the hyperparameter space.
Tuner = kt.BayesianOptimization(
  DeepSetNeuralNetwork,
  objective = "val_loss",
  overwrite = False,
  max_trials = 200,
  directory = '/home/physics/phujdj/DeepLearningParticlePhysics',
  project_name = "DeepSetHyperTrainingMSE",
)


# In[51]:


# Introduce early_stopping to prevent overfitting
early_stopping = callbacks.EarlyStopping(
    min_delta=0.00001,  # The minimum amount of change to count as an improvement
    patience=20,  # The number of epochs to wait before stopping
    restore_best_weights=True,  # Keep the best weights
)
# Prevent spikes in the validation and training loss due to the gradient descent kicking the network out of a local minima
reduce_learn_on_plateau = callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.80, patience=8, min_lr=2e-7)


# In[50]:


# Search the parameter space to obtain the best hyperparameter values
Tuner.search(X_train, y_train, validation_data=(
    X_valid, y_valid), epochs=EPOCHS, callbacks=[early_stopping,reduce_learn_on_plateau])


# In[ ]:

#prints the best parameters obtained from the tuner.
best_hps = Tuner.get_best_hyperparameters(num_trials=100)[0]
print(f"""
The hyperparameter search is complete. The optimal number of track layers is {best_hps.get('track_layers')}, the optimal number of jet layers is {best_hps.get('jet_layers')}, the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}, the optimal dropout rate is {best_hps.get('dropout')} and finally the optimal activation function is {best_hps.get('act_func')}
""")

