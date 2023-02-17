#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
from keras import callbacks
import keras
import DeepSetNeuralNetArchitecture as DSNNA
import uproot
import awkward as ak
import sklearn as sk
import seaborn as sns
from numpy.lib.recfunctions import structured_to_unstructured
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import tensorflow_addons as tfa
import pandas as pd
from hffrag import fixedbinning
from hffrag import binneddensity
from keras.utils.vis_utils import plot_model
from timeit import default_timer as timer
import matplotlib.pyplot as plt


# In[ ]:


# The data is being stored in a tree datastructure.
# We access the charm root using this command
tree = uproot.open("/storage/epp2/phswmv/data/hffrag/hffrag.root:CharmAnalysis")


root_logdir = os.path.join(os.curdir,"my_logs")

def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)


# In[ ]:


# Initial parameters
MASKVAL = -999 # This value is introduced to ensure arrays are regular (Of the same size). They will be masked later by the network
MAXTRACKS = 32 # This value is the maximum number of tracks allowed per event
BATCHSIZE = 32 # This is the batch size of the mini batches used during training
EPOCHS = 1000 # This is the default number of epochs for which the neural network will train providing that early stopping does not occur
MAXEVENTS = 5e5 #This is the maximum number of events that will the program will accept
LR = 1e-4 #This is the default learning rate


# In[ ]:

# Select the features we wish to study
track_features = ["AnalysisTracks_pt", "AnalysisTracks_eta", "AnalysisTracks_phi", "AnalysisTracks_z0sinTheta",
                  "AnalysisTracks_d0sig", "AnalysisTracks_d0", "AnalysisTracks_d0sigPV", "AnalysisTracks_d0PV"]
jet_features = ["AnalysisAntiKt4TruthJets_pt", "AnalysisAntiKt4TruthJets_eta", "AnalysisAntiKt4TruthJets_phi", "AnalysisAntiKt4TruthJets_m",
                "AnalysisAntiKt4TruthJets_ghostB_pt", "AnalysisAntiKt4TruthJets_ghostB_eta","AnalysisAntiKt4TruthJets_ghostB_phi", "AnalysisAntiKt4TruthJets_ghostB_m"]
# In[ ]:


# Read in the data from the root file
features = tree.arrays(jet_features+track_features, entry_stop=MAXEVENTS)


# In[ ]:


# Select the events of interest
events = features[ak.sum(
    features["AnalysisAntiKt4TruthJets_pt"] > 25000, axis=1) > 0]


# In[ ]:


# Displays the number of jets being trained on
jets = events[jet_features][:, 0]
print("The number of jets to train on is: ", len(jets))
print("The number of track features is: ",len(track_features))


# In[ ]:


# Select tracks from the events
tracks = events[track_features]

# Match the tracks to the jets
matchedtracks = tracks[DSNNA.Match_Tracks(jets, tracks)]

# Pad and Flatten the data
matchedtracks = DSNNA.flatten(matchedtracks, MAXTRACKS)


# In[ ]:


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


# In[ ]:

# In[ ]:


# Transform the jet and tracks to unstructed data.
jets = structured_to_unstructured(jets[jet_features[:-4]])
matchedtracks = structured_to_unstructured(matchedtracks)
print(np.shape(jets))


# In[ ]:



# In[ ]:

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


# In[ ]:
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
for bhads_target_feature in range(np.shape(bhads_targets)[1]):
    Bhads_targets = bhads_targets[:,bhads_target_feature]
    mean, std = np.mean(Bhads_targets), np.std(Bhads_targets)
    means = np.append(means,mean)
    stds = np.append(stds,std)
    Standardized_Bhads_targets = (Bhads_targets - mean)/(std)
    Standardized_Bhads_targets = Standardized_Bhads_targets.reshape(-1,1)
    lister.append(Standardized_Bhads_targets)
Standardized_Bhads_targets = np.concatenate(lister,axis = 1)
print(Standardized_Bhads_targets.shape)
print(means,stds)

#Check that this is all the correct shape
print(np.shape(tracks))
print(np.shape(bhads))
print(tracks[0,0])
print(bhads[0])


Tracks_input = np.concatenate([tracks_scaled, Track_fractions, Log_tracks], axis = -1)
print(Tracks_input.shape)

b_jets_input = np.concatenate([b_jets_scaled, Tracks_projection, Sum_Tracks_projection.reshape(-1,1)], axis = -1)
print(b_jets_input.shape)



# In[ ]:
# Split the data into training and validation sets.
X_train, X_valid, y_train, y_valid = train_test_split(
    Tracks_input, Standardized_Bhads_targets, train_size=0.9, random_state = 42)


# In[ ]:


#Cyclical Learning Rate Scheduler:
steps_per_epoch = len(X_train)
clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate = 1e-4,
maximal_learning_rate = 1e-3,
scale_fn = lambda x: 0.90 ** x,
step_size = 2.0 * steps_per_epoch
)
optimizer = tf.keras.optimizers.Nadam(LR,clipnorm = 1.0)

# Builds the deep neural network
track_layers = [64,64,64,64,64]
jet_layers = [128,128,128,128,128]
DeepNet = DSNNA.DeepSetNeuralNetwork(
    [18]+track_layers, jet_layers,4,optimizer)


# In[ ]:


#Summarises the Neural Network Architecture
DeepNet.summary()


# In[ ]:


plot_model(DeepNet, to_file ="NetworkArchitecture.png", show_shapes = True, show_layer_names = True)


# In[ ]:


#Check for the of the training and validation sets
print(np.shape(X_train), np.shape(y_train))


# In[ ]:


class TimingCallback(keras.callbacks.Callback):
    def __init__(self, logs = {}):
        self.logs = []
    def on_epoch_begin(self, epoch, logs ={}):
        self.starttime = timer()
    def on_epoch_end(self, epoch, logs = {}):
        self.logs.append(timer() - self.starttime)


# In[ ]:


# Introduce early_stopping to prevent overfitting
early_stopping = callbacks.EarlyStopping(
    min_delta=0.00001,  # The minimum amount of change to count as an improvement
    patience=30,  # The number of epochs to wait before stopping
    restore_best_weights=True,  # Keep the best weights
)
# Prevent spikes in the validation and training loss due to the gradient descent kicking the network out of a local minima
reduce_learn_on_plateau = callbacks.ReduceLROnPlateau(
    monitor='loss', factor=0.95, patience=10, min_lr=1e-7)

# Save the weights of the model to allow reuse in future.
path = "/home/physics/phujdj/DeepLearningParticlePhysics/CheckPoints/DeepNetWeights&Biases.ckpt"
checkpoint_dir = os.path.dirname(path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=path,
                                                 save_weights_only=True, verbose=0, save_freq = 100 )
#Timer
cb = TimingCallback()

# Learning Scheduler:
exponential_decay_fn = DSNNA.expontial_decay(lr0 = LR,s = 30)
learning_scheduler = tf.keras.callbacks.LearningRateScheduler(exponential_decay_fn)

#Tensorboard Scheduler:
run_logdir = get_run_logdir()
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)


# In[ ]:


steps = np.arange(0,EPOCHS * steps_per_epoch)
lr = clr(steps)
#plt.plot(steps,lr)
#plt.xlabel("Steps")
#plt.ylabel("Learning Rate")
#plt.savefig("/home/physics/phujdj/DeepLearningParticlePhysics/CyclicalLearningRate.png")

X_train_event_sample, y_train_event_sample = X_train[:2], y_train[:2]
X_valid_event_sample, y_valid_event_sample = X_valid[:2], y_train[:2]
print(np.shape(X_train),np.shape(y_train))
print(np.shape(X_train_event_sample),np.shape(y_train_event_sample))

# In[ ]:


# Train the neural network
history = DeepNet.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=BATCHSIZE,
    epochs=EPOCHS,
    callbacks=[cp_callback,cb,reduce_learn_on_plateau]  # Enter call back
)

"""
for index in range(EPOCHS):
  print("epoch number:", index)
  DeepNet.fit(X_train_event_sample, y_train_event_sample, batch_size=BATCHSIZE, callbacks = [cb, reduce_learn_on_plateau,tensorboard_cb])
  pred = DeepNet.predict(X_train_event_sample)
  print(pred - y_train_event_sample)
"""

# In[ ]:

# Plot the loss and validation curves vs epoch
history_df = pd.DataFrame(history.history)
fig,ax = plt.subplots()
ax = np.log(history_df.loc[:, ['loss', 'val_loss']]).plot()
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/HffragLossCurveMSEULTRA3.png')
history_df.to_csv('/home/physics/phujdj/DeepLearningParticlePhysics/HffraghistoryMSEULTRA3.csv')


# In[ ]:

# Output to the console the minimum epoch
print("Minimum validation loss: {}".format(history_df["loss"].min()))


print("Time taken to run Neural Network", sum(cb.logs))


#Evaluate the entire performance of the model
loss = DeepNet.evaluate((Tracks_input,b_jets_input),Standardized_Bhads_targets,verbose = 2)
print("The Loaded DeepNet has loss: ", loss)

Predictions = DeepNet.predict((Tracks_input, b_jets_input))
print(Predictions.shape)


# In[ ]:

Error_px = Standardized_Bhads_targets[:,0] - Predictions[:,0]
Pull_bhads_px = Error_px/np.std(Standardized_Bhads_targets[:,0])

Error_py = Standardized_Bhads_targets[:,1] - Predictions[:,1]
Pull_bhads_py = Error_py/np.std(Standardized_Bhads_targets[:,1])

Error_pz = Standardized_Bhads_targets[:,2] - Predictions[:,2]
Pull_bhads_pz = Error_pz/np.std(Standardized_Bhads_targets[:,2])

Error_projection = Standardized_Bhads_targets[:,3] - Predictions[:,3]
Pull_bhads_projection = Error_projection/np.std(Standardized_Bhads_targets[:,3])


"""
# Output to the console the minimum epoch
print("Minimum loss: {}".format(history_df["loss"].min()))
"""

# In[ ]:

print("Error_px")
print(np.mean(Error_px))
print(np.std(Error_px))
fig = binneddensity(Error_px, fixedbinning(-1,1,100),xlabel = "Error_px")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/Error_Px_scaled_HffragDeepNetUltra3.png')
plt.close()
print("\n")

print("Error_py")
print(np.mean(Error_py))
print(np.std(Error_py))
fig = binneddensity(Error_py, fixedbinning(-1,1,100),xlabel = "Error_py")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/Error_PyDeepNetUltra3.png')
plt.close()
print("\n")

print("Error_pz")
print(np.mean(Error_pz))
print(np.std(Error_pz))
fig = binneddensity(Error_pz, fixedbinning(-1,1,100),xlabel = "Error_pz")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/Error_PzDeepNetMSE4.png')
plt.close()
print("\n")


print("Erorr_projection")
print(np.mean(Error_projection))
print(np.std(Error_projection))
fig = binneddensity(Error_projection, fixedbinning(-1,1,100),xlabel = "Error_projection")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/Error_projectionDeepNetMSE4.png')
plt.close()
print("\n")


print("Pulls_bhads_px")
print(np.mean(Pull_bhads_px))
print(np.std(Pull_bhads_px))
fig = binneddensity(Pull_bhads_px, fixedbinning(-1,1,100),xlabel = "Pull_px")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/Pull_Bhads_PxHffragDeepNetMSE4.png')
plt.close()
print("\n")


print("Pulls_bhads_py")
print(np.mean(Pull_bhads_py))
print(np.std(Pull_bhads_py))
fig = binneddensity(Pull_bhads_py, fixedbinning(-1,1,100),xlabel = "Pull_py")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/Pull_Bhads_PyHffragDeepNetMSE4.png')
plt.close()
print("\n")


print("Pulls_bhads_pz")
print(np.mean(Pull_bhads_pz))
print(np.std(Pull_bhads_pz))
fig = binneddensity(Pull_bhads_pz, fixedbinning(-1,1,100),xlabel = "Pull_pz")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/Pull_Bhads_PzHffragDeepNetMSE4.png')
plt.close()

fig,ax = plt.subplots(figsize = (12,12))
sns.scatterplot(
    y = Predictions[:,0],
    x = Standardized_Bhads_targets[:,0],
    color = "blue"
)
ax.set_title("Scatterplot of the true vs pred X momenta")
ax.set_xlabel("The true X momenta fraction of the tracks from each event")
ax.set_ylabel("The predicted X momenta fraction of b hadron jets")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/ScatterplotPx_Scaled_HffragDeepNetMSEUltra4.png')
plt.close()

fig,ax = plt.subplots(figsize = (12,12))
sns.scatterplot(
    y = Predictions[:,1],
    x = Standardized_Bhads_targets[:,1],
    color = "blue"
)
ax.set_title("Scatterplot of the true vs pred Y momenta")
ax.set_xlabel("The true Y momenta fraction of the tracks from each event")
ax.set_ylabel("The predicted Y momenta fraction of b hadron jets")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/ScatterplotPyHffragDeepNetMSEUltra4.png')
plt.close()


fig,ax = plt.subplots(figsize = (12,12))
sns.scatterplot(
    y = Predictions[:,2],
    x = Standardized_Bhads_targets[:,2],
    color = "blue"
)
ax.set_title("Scatterplot of the true vs pred Z momenta")
ax.set_xlabel("The true Z momenta fraction of the tracks from each event")
ax.set_ylabel("The predicted Z momenta fraction of b hadron jets")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/ScatterplotPzHffragDeepNetMSEUltra4.png')
plt.close()

fig,ax = plt.subplots(figsize = (12,12))
sns.scatterplot(
     y = Predictions[:,3],
    x = Standardized_Bhads_targets[:,3],
    color = "orange"
)
ax.set_title("Scatterplot of the true vs pred Z momenta")
ax.set_xlabel("The true Z momenta of the tracks from each event")
ax.set_ylabel("The predicted Z momenta of b hadron jets")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/ScatterplotProjectionHffragDeepNetMSE4.png')
plt.close()

#Evaluate the entire performance of the model
prediction = DeepNet.predict(X_train_event_sample)
print(prediction,y_train_event_sample)
loss = DeepNet.evaluate(X_train_event_sample,y_train_event_sample,verbose = 2)
print("The Loaded DeepNet has loss: ", loss)

