#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import seaborn as sns
import wandb
#from wandb.keras import WandbCallback
from keras import callbacks
import keras
import DeepSetNeuralNetArchitecture as DSNNA
import tensorflow as tf
import keras_tuner as kt
import keras.layers as layers
from keras import regularizers
from Sum import Sum
import uproot
import awkward as ak
from numpy.lib.recfunctions import structured_to_unstructured
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

plt.style.use("default")
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rc('text',usetex = False)
plt.rc('font',family = 'Times New Roman')

# The data is being stored in a tree datastructure.
# We access the charm root using this command
tree = uproot.open("/storage/epp2/phswmv/data/hffrag/hffrag.root:CharmAnalysis")

# Initial parameters
MASKVAL = -999 # This value is introduced to ensure arrays are regular (Of the same size). They will be masked later by the network
MAXTRACKS = 32 # This value is the maximum number of tracks allowed per event
BATCHSIZE = 128 # This is the batch size of the mini batches used during training
EPOCHS = 800 # This is the default number of epochs for which the neural network will train providing that early stopping does not occur
MAXEVENTS = 2e5 #This is the maximum number of events that will the program will accept
LR = 1e-4 #This is the default learning rate

# Select the features we wish to study
track_features = ["AnalysisTracks_pt", "AnalysisTracks_eta", "AnalysisTracks_phi", "AnalysisTracks_z0sinTheta",
                  "AnalysisTracks_d0sig", "AnalysisTracks_d0", "AnalysisTracks_d0sigPV", "AnalysisTracks_d0PV"]
jet_features = ["AnalysisAntiKt4TruthJets_pt", "AnalysisAntiKt4TruthJets_eta", "AnalysisAntiKt4TruthJets_phi", "AnalysisAntiKt4TruthJets_m",
                "AnalysisAntiKt4TruthJets_ghostB_pt", "AnalysisAntiKt4TruthJets_ghostB_eta","AnalysisAntiKt4TruthJets_ghostB_phi", "AnalysisAntiKt4TruthJets_ghostB_m"]

# Read in the dat from the root file
features = tree.arrays(jet_features+track_features, entry_stop=MAXEVENTS)

# Select the events of interest
events = features[ak.sum(
    features["AnalysisAntiKt4TruthJets_pt"] > 25000, axis=1) > 0]

# Displays the number of jets being trained on
jets = events[jet_features][:, 0]
print("The number of jets to train on is: ", len(jets))
print("The number of track features is: ",len(track_features))

# Select tracks from the events
tracks = events[track_features]

# Match the tracks to the jets
matchedtracks = tracks[DSNNA.Match_Tracks(jets, tracks)]

# Pad and Flatten the data
matchedtracks = DSNNA.flatten(matchedtracks, MAXTRACKS)

# Identify the the bottom jets and their associated tracks
bjets = ak.sum(jets["AnalysisAntiKt4TruthJets_ghostB_pt"] > 5000, axis=1) > 0
jets = jets[bjets]

# Obtain the pt, eta and phi of each b hadron jet
bhads_pt = jets["AnalysisAntiKt4TruthJets_ghostB_pt"][:, 0].to_numpy()
bhads_eta = jets["AnalysisAntiKt4TruthJets_ghostB_eta"][:,0].to_numpy()
bhads_phi = jets["AnalysisAntiKt4TruthJets_ghostB_phi"][:,0].to_numpy()
bhads_m = jets["AnalysisAntiKt4TruthJets_ghostB_m"][:,0].to_numpy()

jets_pt = jets["AnalysisAntiKt4TruthJets_pt"].to_numpy()
jets_eta = jets["AnalysisAntiKt4TruthJets_eta"].to_numpy()
jets_phi = jets["AnalysisAntiKt4TruthJets_phi"].to_numpy()
jets_m = jets["AnalysisAntiKt4TruthJets_m"].to_numpy()
b_jets = np.stack([jets_pt,jets_eta,jets_phi, jets_m], axis = -1)

bhads = np.stack([bhads_pt,bhads_eta,bhads_phi, bhads_m],axis = -1) #Combine the momentum, eta and phi for each jet into one array

print("There are {} outputs".format(np.shape(bhads)[1])) # Display the number of target features the neural network will predict
matchedtracks = matchedtracks[bjets]
print("There are {} inputs".format(np.shape(matchedtracks)[1])) # Display the number of target features the neural network will use in it's predictions

# Transform the jet and tracks to unstructed data.
jets = structured_to_unstructured(jets[jet_features[:-4]])
matchedtracks = structured_to_unstructured(matchedtracks)

# Convert the coordinates of the b jets and tracks to cartesian coordinates
polartracks = matchedtracks.to_numpy()
Num_events = MAXTRACKS
tracks_pt = polartracks[:,:,0].reshape(-1,Num_events,1)
tracks_eta = polartracks[:,:,1].reshape(-1,Num_events,1)
tracks_phi = polartracks[:,:,2].reshape(-1,Num_events,1)

tracks_pep = np.concatenate([tracks_pt,tracks_eta,tracks_phi], axis = -1) 
print(tracks_pep.shape)

jets_pt = b_jets[:,0].reshape(-1,1)
jets_eta = b_jets[:,1].reshape(-1,1)
jets_phi = b_jets[:,2].reshape(-1,1)

b_jets_pep = np.concatenate([jets_pt,jets_eta,jets_phi], axis = -1) 
print(b_jets_pep.shape)

tracks_p = DSNNA.pt_eta_phi_2_px_py_pz_tracks(matchedtracks.to_numpy())
bhads = DSNNA.pt_eta_phi_2_px_py_pz_jets(bhads)
b_jets_p = DSNNA.pt_eta_phi_2_px_py_pz_jets(b_jets)
b_jets_m = b_jets[:,-1].reshape(-1,1)

#Combine the momenta of the tracks with the rest of the track features to form the track dataset
tracks = np.concatenate([tracks_p,tracks_pep,matchedtracks[:,:,3:].to_numpy()],axis = 2)
b_jets = np.concatenate([b_jets_p,b_jets_pep,b_jets_m] ,axis = 1)

tracks = np.ma.masked_values(tracks,-999)

bhads_fractions_px = bhads[:,0]/b_jets[:,0]
bhads_fractions_py = bhads[:,1]/b_jets[:,1]
bhads_fractions_pz = bhads[:,2]/b_jets[:,2]
print(bhads_fractions_px.shape)

b_jets_mag = np.linalg.norm(b_jets[:,:3], axis = 1)
bhads_fractions = np.stack([bhads_fractions_px,bhads_fractions_py, bhads_fractions_pz], axis = -1)
bhads_projection = ((bhads[:,:3]*b_jets[:,:3]).sum(axis = 1))/(b_jets_mag**2)
print(bhads_fractions.shape)

print(np.max(bhads_fractions), np.min(bhads_fractions_px))
array = [x for x in range(bhads_fractions_px.shape[0])]
bhads_trial = np.stack([array,bhads_fractions_px, bhads_fractions_py, bhads_fractions_pz, bhads_projection], axis = -1)
bhads_fractions_clean  = bhads_trial[(bhads_trial[:,1] < 1.5) & (bhads_trial[:,1] > 0) & (bhads_trial[:,2] < 1.5) & (bhads_trial[:,2] > 0) & (bhads_trial[:,3] < 1.5) & (bhads_trial[:,3] > 0) & (bhads_trial[:,4] > 0) & (bhads_trial[:,4] <1.5)]
print(bhads_fractions_clean.shape)
print(np.max(bhads_fractions_clean[:,1]), np.min(bhads_fractions_clean[:,1]))
indices = bhads_fractions_clean[:,0]
print(indices.shape)

indices = [int(x) for x in indices]
print(indices[:5])

tracks = tracks[indices]
b_jets = b_jets[indices]
bhads = bhads[indices]
b_jets_m = b_jets_m[indices]
bhads_m = bhads_m[indices]

print(bhads_fractions[0])
print(bhads[0,0]/b_jets[0,0])
print(np.min(bhads_fractions),np.max(bhads_fractions))

print(bhads_fractions_px[0])
print(bhads[0,0]/b_jets[0,0])
print(np.min(bhads_fractions_px),np.max(bhads_fractions_px))

b_jets_mag = np.linalg.norm(b_jets[:,:3], axis = 1)
bhads_mag = np.linalg.norm(bhads[:,:3], axis = 1)
tracks_Momentum = np.sum(np.linalg.norm(tracks[:,:,:3], axis = 2))

bhads_fractions_px = bhads[:,0]/b_jets[:,0]
bhads_fractions_py = bhads[:,1]/b_jets[:,1]
bhads_fractions_pz = bhads[:,2]/b_jets[:,2]
print(bhads_fractions_px.shape)

b_jets_energy = np.sqrt((b_jets_m[:,0]**2) + (b_jets_mag**2))
b_jets_energy.shape

b_jets_energy_pt = np.sqrt((b_jets_m[:,0]**2) + (b_jets[:,4]**2))
b_jets_energy_pt.shape


bhads_energy = np.sqrt((bhads_m**2) + (bhads_mag**2))
bhads_energy.shape

print(bhads_fractions[0])
print(bhads[0,0]/b_jets[0,0])
print(np.min(bhads_fractions),np.max(bhads_fractions))

print(bhads_fractions_px[0])
print(bhads[0,0]/b_jets[0,0])
print(np.min(bhads_fractions_px),np.max(bhads_fractions_px))

print("Hello")
sum_px_tracks = np.sum(tracks[:,:,0], axis = 1)
sum_py_tracks = np.sum(tracks[:,:,1], axis = 1)
sum_pz_tracks = np.sum(tracks[:,:,2], axis = 1)
sum_pt_tracks = np.sum(tracks[:,:,3], axis = 1)
print(sum_pt_tracks.shape)

sum_px_tracks_RSE = np.sqrt(np.sum(tracks[:,:,0]**2, axis = 1))
sum_py_tracks_RSE= np.sqrt(np.sum(tracks[:,:,1]**2, axis = 1))
sum_pz_tracks_RSE = np.sqrt(np.sum(tracks[:,:,2]**2, axis = 1))
sum_pt_tracks_RSE = np.sqrt(np.sum(tracks[:,:,3]**2, axis = 1))
print(sum_pt_tracks_RSE.shape)

RSM_scaled_px = sum_px_tracks_RSE/sum_px_tracks
RSM_scaled_py = sum_py_tracks_RSE/sum_py_tracks
RSM_scaled_pz = sum_pz_tracks_RSE/sum_pz_tracks
RSM_scaled_pt = sum_pt_tracks_RSE/sum_pt_tracks
print(RSM_scaled_pt.shape)

RMS_scaled_px = np.sqrt(np.sum(tracks[:,:,0]**2, axis = 1)/MAXTRACKS)
RMS_scaled_py = np.sqrt(np.sum(tracks[:,:,1]**2, axis = 1)/MAXTRACKS)
RMS_scaled_pz = np.sqrt(np.sum(tracks[:,:,2]**2, axis = 1)/MAXTRACKS)
RMS_scaled_pt = np.sqrt(np.sum(tracks[:,:,3]**2, axis = 1)/MAXTRACKS)
print(RMS_scaled_pt.shape)

Log_px_tracks = np.log(abs(tracks[:,:,0]/b_jets[:,np.newaxis,0]))
Log_py_tracks = np.log(abs(tracks[:,:,1]/b_jets[:,np.newaxis,1]))
Log_pz_tracks = np.log(abs(tracks[:,:,2]/b_jets[:,np.newaxis,2]))
Log_pt_tracks = np.log(abs(tracks[:,:,3]/b_jets[:,np.newaxis,3]))
Log_tracks = np.stack([Log_px_tracks, Log_py_tracks, Log_pz_tracks, Log_pt_tracks], axis = -1)

Log_Sum_px = np.log(sum_px_tracks/b_jets[:,0])
Log_Sum_py = np.log(sum_py_tracks/b_jets[:,1])
Log_Sum_pz = np.log(sum_pz_tracks/b_jets[:,2])
Log_Sum_pt = np.log(sum_pt_tracks/b_jets[:,3])
Log_Momenta = np.log(abs(tracks_Momentum/np.sum(b_jets[:,:3], axis = 1)))
print(Log_Sum_pt.shape)

tracks_fractions_px = tracks[:,:,0]/b_jets[:,np.newaxis,0]
tracks_fractions_py = tracks[:,:,1]/b_jets[:,np.newaxis,1]
tracks_fractions_pz = tracks[:,:,2]/b_jets[:,np.newaxis,2]
tracks_fractions_pt = tracks[:,:,3]/b_jets[:,np.newaxis,3]
print(tracks_fractions_pt.shape)
Track_fractions = np.stack([tracks_fractions_px,tracks_fractions_py, tracks_fractions_pz], axis = -1)
print(Track_fractions.shape)

print(Track_fractions.shape)
print(tracks[0,0,0]/b_jets[0,0])
print(np.mean(Track_fractions),np.std(Track_fractions))

print("hey")
Tracks_projection = ((tracks[:,:,:3]*b_jets[:,np.newaxis,:3]).sum(axis = 2)/(b_jets_mag[:,np.newaxis]**2))
print(Tracks_projection.shape)
Track_Momenta = np.stack([sum_px_tracks, sum_py_tracks, sum_pz_tracks], axis = -1)
print(Track_Momenta.shape)
Sum_Tracks_projection = ((Track_Momenta*b_jets[:,:3]).sum(axis = 1))/(b_jets_mag**2)
print(Sum_Tracks_projection.shape)

b_jet_energy_ratio_px = sum_px_tracks/b_jets_energy
b_jet_energy_ratio_py = sum_py_tracks/b_jets_energy
b_jet_energy_ratio_pz = sum_pz_tracks/b_jets_energy
b_jet_energy_ratio_pt = sum_pt_tracks/b_jets_energy
print(b_jet_energy_ratio_pt.shape)

b_jet_energy_ratio_cart = b_jets_mag/b_jets_energy
b_jet_energy_ratio_pt = b_jets[:,4]/b_jets_energy
print(b_jet_energy_ratio_pt.shape)

b_jet_energy_ratio_total = np.sum(b_jets[:,4])/np.sum(b_jets_energy)  
b_jet_transverse_mass = np.sqrt(b_jets_energy**2 - b_jets[:,2]**2)
print(b_jet_transverse_mass[0])
print(b_jet_energy_ratio_total.shape)
print(b_jet_transverse_mass.shape)
print(np.full((len(b_jets)),b_jet_energy_ratio_total).shape)
print("end")
bhads_projection = ((bhads[:,:3]*b_jets[:,:3]).sum(axis = 1))/(b_jets_mag**2)

print(np.mean(b_jets_energy),np.std(b_jets_energy))
b_jets_energy = (b_jets_energy - np.mean(b_jets_energy))/(np.std(b_jets_energy))

print(np.mean(bhads_energy),np.std(bhads_energy))
bhads_energy = (bhads_energy - np.mean(bhads_energy))/(np.std(bhads_energy))

b_jets = np.stack([b_jets[:,0], b_jets[:,1], b_jets[:,2],b_jets[:,3],b_jets[:,4], b_jets[:,5], b_jets[:,6], b_jets_mag, sum_px_tracks, sum_py_tracks, sum_pz_tracks, sum_pt_tracks, sum_px_tracks_RSE, sum_py_tracks_RSE, sum_pz_tracks_RSE, sum_pt_tracks_RSE, RSM_scaled_px, RSM_scaled_py, RSM_scaled_pz, RSM_scaled_pt, RMS_scaled_px, RMS_scaled_py, RMS_scaled_pz, RMS_scaled_pt, b_jet_transverse_mass, Log_Sum_px, Log_Sum_py, Log_Sum_pz, Log_Sum_pt, Log_Momenta, b_jets_energy, b_jet_energy_ratio_px, b_jet_energy_ratio_py, b_jet_energy_ratio_pz, b_jet_energy_ratio_cart, b_jet_energy_ratio_pt, np.full((len(b_jets),),b_jet_energy_ratio_total)], axis = -1)
bhads_targets = np.stack([bhads_fractions_px,bhads_fractions_py, bhads_fractions_pz, bhads_projection], axis = -1)

Scaler_tracks = StandardScaler()
Num_events,Num_tracks,Num_features = np.shape(tracks)
Scaled_tracks = np.reshape(tracks, newshape=(-1,Num_features))
tracks_scaled = Scaler_tracks.fit_transform(Scaled_tracks)
tracks_scaled = np.reshape(tracks_scaled, newshape= (Num_events,Num_tracks,Num_features))
print(np.shape(tracks_scaled))

Scaler_jets = StandardScaler()
Num_events,Num_features = np.shape(b_jets)
b_jets_scaled = np.reshape(b_jets, newshape=(-1,Num_features))
b_jets_scaled = Scaler_jets.fit_transform(b_jets_scaled)
b_jets_scaled = np.reshape(b_jets_scaled, newshape= (Num_events,Num_features))
print(np.shape(b_jets_scaled))

means = []
stds = []
lister = []
for bhads_feature in range(np.shape(bhads)[1]):
    Bhads = bhads[:,bhads_feature]
    mean, std = np.mean(Bhads), np.std(Bhads)
    means = np.append(means,mean)
    stds = np.append(stds,std)
    Standardized_Bhads = (Bhads - mean)/(std)
    Standardized_Bhads = Standardized_Bhads.reshape(-1,1)
    lister.append(Standardized_Bhads)
Standardized_Bhads = np.concatenate(lister,axis = 1)
print(Standardized_Bhads.shape)
print(means,stds)

Tracks_input = np.concatenate([tracks_scaled, Track_fractions, Log_tracks], axis = -1)
print(Tracks_input.shape)

b_jets_input = np.concatenate([b_jets_scaled, Tracks_projection, Sum_Tracks_projection.reshape(-1,1)], axis = -1)
print(b_jets_input.shape)

class TimeDistributedResidualUnits(keras.layers.Layer):
    def __init__(self, d_model, regularizer_strength = 0.01, dropout = 0.001):
        super().__init__()
        self.seq = tf.keras.Sequential([
            layers.TimeDistributed(layers.Dense(d_model, activation = "gelu",  kernel_initializer = "he_normal", kernel_regularizer= regularizers.l1_l2(regularizer_strength))),
            layers.TimeDistributed(layers.Dense(d_model, activation = "gelu",  kernel_initializer = "he_normal", kernel_regularizer= regularizers.l1_l2(regularizer_strength))),
            layers.Dropout(dropout)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x

class ResidualUnits(keras.layers.Layer):
    def __init__(self, d_model, regularizer_strength = 0.01, dropout = 0.001):
        super().__init__()
        self.seq = tf.keras.Sequential([
            layers.Dense(d_model, activation = "gelu",  kernel_initializer = "he_normal", kernel_regularizer= regularizers.l1_l2(regularizer_strength)),
            layers.Dense(d_model, activation = "gelu",  kernel_initializer = "he_normal", kernel_regularizer= regularizers.l1_l2(regularizer_strength)),
            layers.Dropout(dropout)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x

def DeepSetsProjection(hp):
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
    n_features  = np.shape(b_jets_input)[1]
    Dropout_rate = hp.Choice('dropout_rate', [0.1,0.01,0.001,0.0001,0.00001])
    track_layer_Neurons = hp.Choice('track_layers_Num_Neurons',[16,32,64,128,256,512])
    jet_layer_Neurons = hp.Choice('jet_layers_Number_Neurons',[16,32,64,128,256,512])
    b_jet_layer_Neurons = hp.Choice('jet_layers_Number_Neurons',[16,32,64,128,256,512])
    activation_func = hp.Choice('act_func',["relu","elu","selu", "gelu"])
    Num_tracks_layer = hp.Choice('Num_track_layers_Neurons',[1,2,3,4,5,6])
    Num_jets_layer = hp.Choice('Num_jet_layers_Neurons',[1,2,3,4,5,6])
    Num_b_jets_layer = hp.Choice('Num_jet_layers_Neurons',[1,2,3,4,5,6])
    Learning_rate = hp.Choice('learning_rate',[1e-2,1e-3,1e-4,1e-5])
    Initializer_value = hp.Choice('Initalizers', ["he_normal", "lecun_normal","glorot_uniform"])
    regularizer_strength = hp.Choice("regularizer", [1e-6,1e-5,1e-4,1e-3,1e-2,1e-1])
    beta_1 = hp.Choice("beta_1", [0.9,0.95,0.98,0.999])
    beta_2 = hp.Float("beta_2", [0.9,0.,95,0.98,0,999])
    weight_decay = hp.Choice("Weight_Decay", [0.1,0.2,0.4,0.8,1.0])

    #Create the track and jet layers
    track_layers = [len(track_features)]+[track_layer_Neurons for x in range(Num_tracks_layer)]
    jet_layers  = [jet_layer_Neurons for x in range(Num_jets_layer)]
    b_jet_layers = [b_jet_layer_Neurons for x in range(Num_b_jets_layer)]

    tracks = layers.Input(shape = (None, track_layers[0]))
    jets = layers.Input(shape = (n_features))

    #Set the number of targets being explored
    n_targets = 4
    
    outputs_tracks = tracks
    outputs_jets = jets

    for nodes in track_layers[:-1]:
        outputs_tracks = layers.Dense(nodes, activation = activation_func, kernel_initializer=Initializer_value)(outputs_tracks)
        outputs_tracks = TimeDistributedResidualUnits(nodes,regularizer_strength, Dropout_rate)(outputs_tracks)

    outputs_tracks = layers.TimeDistributed(layers.Dense( 
    track_layers[-1], activation = activation_func))(outputs_tracks) # Apply softmax to ouput the results of the track neural network as probabilities
    outputs_tracks = Sum()(outputs_tracks) # Sum the outputs to make use of permutation invariance
    
    for nodes in b_jet_layers[:-1]:
        outputs_jets = layers.Dense(nodes, activation=activation_func, kernel_initializer= Initializer_value)(outputs_jets) # We use relu and the corresponding he_normal for the activation function and bias initializer

        outputs_jets = layers.LayerNormalization()(outputs_jets)
    outputs_jets = layers.Dense(b_jet_layers[-1], activation = activation_func, kernel_initializer=Initializer_value, kernel_regularizer=regularizers.l1_l2(regularizer_strength))(outputs_jets) # Apply softmax to ouput the results of the track neural network as probabilities

    outputs = layers.concatenate([outputs_tracks, outputs_jets], axis = -1)

    for nodes in jet_layers[:-1]:
        outputs = layers.Dense(nodes, activation = activation_func, kernel_initializer=Initializer_value, kernel_regularizer= regularizers.l1_l2(regularizer_strength))(outputs)
        outputs = ResidualUnits(nodes,regularizer_strength, Dropout_rate)(outputs)

    outputs_Target_jets = layers.Dense(n_targets, activation='linear')(outputs)
    
    Model = keras.Model(inputs = [tracks,jets], outputs = outputs_Target_jets)

    Model.compile(
            optimizer = tf.keras.optimizers.Nadam(learning_rate=Learning_rate, beta_1 = beta_1, beta_2 = beta_2, decay = weight_decay),
            loss = tf.keras.losses.MSE,
            jit_compile = True
        )
    return Model
    
# Split the data into training and validation sets.
X_train, X_valid, y_train, y_valid = train_test_split(
    Tracks_input, Standardized_Bhads, train_size=0.9, random_state = 42)

# Split the data into training and validation sets.
X_train_b_jets, X_valid_b_jets, y_train_b_jets, y_valid_b_jets = train_test_split(
    b_jets_input, Standardized_Bhads, train_size=0.9, random_state = 42)

# Introduce early_stopping to prevent overfitting
early_stopping = callbacks.EarlyStopping(
    min_delta=0.00001,  # The minimum amount of change to count as an improvement
    patience=50,  # The number of epochs to wait before stopping
    restore_best_weights=True,  # Keep the best weights
)
# Prevent spikes in the validation and training loss due to the gradient descent kicking the network out of a local minima
reduce_learn_on_plateau = callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.90, patience=15, min_lr=1e-9)

#Set up the hyperparameter
SEED = tf.random.set_seed(42) # Generate a random seed

Tuner = kt.BayesianOptimization(
  DeepSetsProjection,
  objective = "val_loss",
  overwrite = True,
  max_trials = 200,
  directory = '/home/physics/phujdj/DeepLearningParticlePhysics',
  project_name = "DeepSetProjector",
)

# Search the parameter space to obtain the best hyperparameter values
Tuner.search((X_train,X_train_b_jets), y_train, validation_data=(
    (X_valid,X_valid_b_jets), y_valid), epochs=40, callbacks=[early_stopping, reduce_learn_on_plateau])

best_hps = Tuner.get_best_hyperparameters(num_trials=200)[0]
print(f"""
The hyperparameter search is complete. The optimal number of track layers is {best_hps.get('track_layers')}, the optimal number of jet layers is {best_hps.get('jet_layers')}, the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}, the optimal dropout rate is {best_hps.get('dropout')} and finally the optimal activation function is {best_hps.get('act_func')}
""")