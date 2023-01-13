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
from numpy.lib.recfunctions import structured_to_unstructured
from sklearn.model_selection import train_test_split
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
tree = uproot.open("hffrag.root:CharmAnalysis")
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
EPOCHS = 3000 # This is the default number of epochs for which the neural network will train providing that early stopping does not occur
MAXEVENTS = 1e20 #This is the maximum number of events that will the program will accept
LR = 0.01 #This is the default learning rate


# In[ ]:


# Select the features we wish to study
track_features = ["AnalysisTracks_pt", "AnalysisTracks_eta", "AnalysisTracks_phi", "AnalysisTracks_z0sinTheta",
                  "AnalysisTracks_d0sig", "AnalysisTracks_d0", "AnalysisTracks_d0sigPV", "AnalysisTracks_d0PV"]
jet_features = ["AnalysisAntiKt4TruthJets_pt", "AnalysisAntiKt4TruthJets_eta", "AnalysisAntiKt4TruthJets_phi",
                "AnalysisAntiKt4TruthJets_ghostB_pt", "AnalysisAntiKt4TruthJets_ghostB_eta","AnalysisAntiKt4TruthJets_ghostB_phi"]


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

bhads = np.stack([bhads_pt,bhads_eta,bhads_phi],axis = -1) #Combine the momentum, eta and phi for each jet into one array

print("There are {} outputs".format(np.shape(bhads)[1])) # Display the number of target features the neural network will predict
matchedtracks = matchedtracks[bjets]
print("There are {} inputs".format(np.shape(matchedtracks)[1])) # Display the number of target features the neural network will use in it's ppredictions


# In[ ]:


print(np.shape(bhads)) #Check the shape of the neural network
print(np.shape(jet_features[:-1])) #Check for shape of the jet features
print(jets[jet_features[0]]) # Check the jets


# In[ ]:


# Transform the jet and tracks to unstructed data.
jets = structured_to_unstructured(jets[jet_features[:-3]])
matchedtracks = structured_to_unstructured(matchedtracks)
print(np.shape(jets))


# In[ ]:


#Check the matchtracks are the correct shape
print(matchedtracks[:, 0:1])
print(np.shape(matchedtracks[:, :, 3]))


# In[ ]:


# Convert the coordinates of the b jets and tracks to cartesian coordinates
tracks_p = DSNNA.pt_eta_phi_2_px_py_pz_tracks(matchedtracks.to_numpy())
bhads = DSNNA.pt_eta_phi_2_px_py_pz_jets(bhads)

#Check the shape of the momenta of the tracks and the rest of the data is consistent
print(np.shape(tracks_p))
print(np.shape(matchedtracks[:, :, 3:]))

#Combine the momenta of the tracks with the rest of the track features to form the track dataset
tracks = np.concatenate([tracks_p,matchedtracks[:,:,3:].to_numpy()],axis = 2)


# In[ ]:


#Check that this is all the correct shape
print(np.shape(tracks))
print(np.shape(bhads))
print(tracks[0,0])
print(bhads[0])


# In[ ]:


# Split the data into training and validation sets.
X_train, X_valid, y_train, y_valid = train_test_split(
    tracks, bhads, train_size=0.7, random_state=42)
#Save the training and validation datasets.
np.save("/home/physics/phujdj/DeepLearningParticlePhysics/TrainingAndValidationData/X_train_data.npy",X_train)
np.save("/home/physics/phujdj/DeepLearningParticlePhysics/TrainingAndValidationData/X_valid_data.npy",X_valid)
np.save("/home/physics/phujdj/DeepLearningParticlePhysics/TrainingAndValidationData/y_train_data.npy",y_train)
np.save("/home/physics/phujdj/DeepLearningParticlePhysics/TrainingAndValidationData/y_valid_data.npy",y_valid)


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
track_layers = [256,256,256,256,256]
jet_layers = [512,512,512,512,512]
DeepNet = DSNNA.DeepSetNeuralNetwork(
    [len(track_features)]+track_layers, jet_layers,3,optimizer)


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
history_df.to_csv('/home/physics/phujdj/DeepLearningParticlePhysics/history.csv')


# In[ ]:


print("Time taken to run Neural Network", sum(cb.logs))


# In[ ]:

"""
# Output to the console the minimum epoch
print("Minimum loss: {}".format(history_df["loss"].min()))
"""

# In[ ]:


#Evaluate the entire performance of the model
prediction = DeepNet.predict(X_train_event_sample)
print(prediction,y_train_event_sample)
loss = DeepNet.evaluate(X_train_event_sample,y_train_event_sample,verbose = 2)
print("The Loaded DeepNet has loss: ", loss)

