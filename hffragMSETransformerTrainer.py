"""
*Filename: hffragMSETransformerTrainer
*Description: This python file is used to train the transformer architecture using a
*MSE loss to regress the bhadron targets. It uses the MSE version of the hffragTransformer.
*Once trained the program returns the resolutions plots and scatterplots of the true vs the
*predicted.

Date: 16/02/2023
Author: Kaylen Smith Darnbrook
"""

#!/usr/bin/env python
# coding: utf-8

# In[1]:

#Starts by importing the relevant python modules
import os
import numpy as np
from keras import callbacks
import keras
import DeepSetNeuralNetArchitecture as DSNNA
import hffragTransformerJetMSEMultivariate as hffragT
import uproot
import awkward as ak
from numpy.lib.recfunctions import structured_to_unstructured
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns
from hffrag import fixedbinning
from hffrag import binneddensity
from timeit import default_timer as timer
import matplotlib.pyplot as plt


# In[2]:

# In[3]:


# The data is being stored in a tree datastructure.
# We access the charm root using this command
tree = uproot.open(
    "/storage/epp2/phswmv/data/hffrag/hffrag.root:CharmAnalysis")


# In[4]:


# Initial parameters
# This value is introduced to ensure arrays are regular (Of the same size). They will be masked later by the network
MASKVAL = -999
MAXTRACKS = 32  # This value is the maximum number of tracks allowed per event
BATCHSIZE = 128  # This is the batch size of the mini batches used during training
EPOCHS = 1  # This is the default number of epochs for which the neural network will train providing that early stopping does not occur
MAXEVENTS = 5e5  # This is the maximum number of events that will the program will accept
LR = 1e-4  # This is the default learning rate
num_layers = 5
d_model = 64
dff = 64
num_heads = 5
n_targets = 1
dropout_rate = 0.0001
track_layers = 5
jet_layers = [128, 128, 128, 128]


# In[5]:

# In[6]:


# Select the features we wish to study
track_features = ["AnalysisTracks_pt", "AnalysisTracks_eta", "AnalysisTracks_phi", "AnalysisTracks_z0sinTheta",
                  "AnalysisTracks_d0sig", "AnalysisTracks_d0", "AnalysisTracks_d0sigPV", "AnalysisTracks_d0PV"]
jet_features = ["AnalysisAntiKt4TruthJets_pt", "AnalysisAntiKt4TruthJets_eta", "AnalysisAntiKt4TruthJets_phi",
                "AnalysisAntiKt4TruthJets_ghostB_pt", "AnalysisAntiKt4TruthJets_ghostB_eta", "AnalysisAntiKt4TruthJets_ghostB_phi"]

# In[7]:


# Read in the dat from the root file
features = tree.arrays(jet_features+track_features, entry_stop=MAXEVENTS)

# In[8]:


# Select the events of interest
events = features[ak.sum(
    features["AnalysisAntiKt4TruthJets_pt"] > 25000, axis=1) > 0]


# In[9]:


# Displays the number of jets being trained on
jets = events[jet_features][:, 0]
print("The number of jets to train on is: ", len(jets))
print("The number of track features is: ", len(track_features))

# In[10]:


# Select tracks from the events
tracks = events[track_features]

# Match the tracks to the jets
matchedtracks = tracks[DSNNA.Match_Tracks(jets, tracks)]

# Pad and Flatten the data
matchedtracks = DSNNA.flatten(matchedtracks, MAXTRACKS)


# In[11]:


# Identify the the bottom jets and their associated tracks
bjets = ak.sum(jets["AnalysisAntiKt4TruthJets_ghostB_pt"] > 5000, axis=1) > 0
jets = jets[bjets]

# Obtain the pt, eta and phi of each b hadron jet
bhads_pt = jets["AnalysisAntiKt4TruthJets_ghostB_pt"][:, 0].to_numpy()
bhads_eta = jets["AnalysisAntiKt4TruthJets_ghostB_eta"][:, 0].to_numpy()
bhads_phi = jets["AnalysisAntiKt4TruthJets_ghostB_phi"][:, 0].to_numpy()

jets_pt = jets["AnalysisAntiKt4TruthJets_pt"].to_numpy()
jets_eta = jets["AnalysisAntiKt4TruthJets_eta"].to_numpy()
jets_phi = jets["AnalysisAntiKt4TruthJets_phi"].to_numpy()
b_jets = np.stack([jets_pt, jets_eta, jets_phi], axis=-1)

# Combine the momentum, eta and phi for each jet into one array
bhads = np.stack([bhads_pt, bhads_eta, bhads_phi], axis=-1)

# Display the number of target features the neural network will predict
print("There are {} outputs".format(np.shape(bhads)[1]))
matchedtracks = matchedtracks[bjets]
# Display the number of target features the neural network will use in it's ppredictions
print("There are {} inputs".format(np.shape(matchedtracks)[1]))


# In[12]:


# Transform the jet and tracks to unstructed data.
jets = structured_to_unstructured(jets[jet_features[:-3]])
matchedtracks = structured_to_unstructured(matchedtracks)


# In[13]:


# Convert the coordinates of the b jets and tracks to cartesian coordinates
tracks_p = DSNNA.pt_eta_phi_2_px_py_pz_tracks(matchedtracks.to_numpy())
bhads = DSNNA.pt_eta_phi_2_px_py_pz_jets(bhads)
b_jets = DSNNA.pt_eta_phi_2_px_py_pz_jets(b_jets)

# Combine the momenta of the tracks with the rest of the track features to form the track dataset
tracks = np.concatenate([tracks_p, matchedtracks[:, :, 3:].to_numpy()], axis=2)


# In[14]:
#Generate the derived features which aid in convergence
b_jets_mag = np.linalg.norm(b_jets, axis=1)
bhads_scaled_px = bhads[:, 0]/b_jets_mag
bhads_scaled_py = bhads[:, 1]/b_jets_mag
bhads_scaled_pz = bhads[:, 2]/b_jets_mag
print(bhads_scaled_px.shape)

bhads_scaled = np.stack(
    [bhads_scaled_px, bhads_scaled_py, bhads_scaled_pz], axis=-1)
print(bhads_scaled.shape)

bhads_fractions_px = bhads[:, 0]/b_jets[:, 0]
bhads_fractions_py = bhads[:, 1]/b_jets[:, 1]
bhads_fractions_pz = bhads[:, 2]/b_jets[:, 2]
print(bhads_scaled_px.shape)

bhads_fractions = np.stack(
    [bhads_fractions_px, bhads_fractions_py, bhads_fractions_pz], axis=-1)
print(bhads_fractions.shape)

print(bhads_fractions[0])
print(bhads[0, 0]/b_jets[0, 0])
print(np.min(bhads_fractions), np.max(bhads_fractions))

print(bhads_scaled[0])
print(bhads[0, 0]/b_jets[0, 0])
print(np.min(bhads_scaled), np.max(bhads_scaled))

sum_px_tracks = np.sum(tracks[:, :, 0], axis=1)
sum_py_tracks = np.sum(tracks[:, :, 1], axis=1)
sum_pz_tracks = np.sum(tracks[:, :, 2], axis=1)

bhads_projection = ((bhads*b_jets).sum(axis=1))/(b_jets_mag**2)

b_jets = np.stack([b_jets[:, 0], b_jets[:, 1], b_jets[:, 2],
                  sum_px_tracks, sum_py_tracks, sum_pz_tracks], axis=-1)

# In[15]:


# In[16]:
# In[17]:


# In[18]:


# In[20]:


# In[22]:

#Standardize the feature and target datasets across the features.
Scaler_tracks = StandardScaler()
Num_events, Num_tracks, Num_features = np.shape(tracks)
Scaled_tracks = np.reshape(tracks, newshape=(-1, Num_features))
tracks_scaled = Scaler_tracks.fit_transform(Scaled_tracks)
tracks_scaled = np.reshape(tracks_scaled, newshape=(
    Num_events, Num_tracks, Num_features))
print(np.shape(tracks_scaled))
print(tracks_scaled[0, 0, :])

Scaler_jets = StandardScaler()
Num_events, Num_features = np.shape(b_jets)
b_jets_scaled = np.reshape(b_jets, newshape=(-1, Num_features))
b_jets_scaled = Scaler_jets.fit_transform(b_jets)
b_jets_scaled = np.reshape(b_jets_scaled, newshape=(Num_events, Num_features))
print(np.shape(b_jets_scaled))
print(b_jets_scaled[0, :])

# Split the data into training and validation sets.
X_train, X_valid, y_train, y_valid = train_test_split(
    tracks_scaled, bhads_scaled, train_size=0.7, random_state=42)


# In[23]:


# Split the data into training and validation sets.
X_train_b_jets, X_valid_b_jets, y_train_b_jets, y_valid_b_jets = train_test_split(
    b_jets_scaled, bhads_scaled, train_size=0.7, random_state=42)


# In[24]:

#Single Event Test Cases.
X_train_event, y_train_event = np.array([X_train[0]]), np.array([y_train[0]])
X_valid_event, y_valid_event = np.array([X_valid[0]]), np.array([y_valid[0]])
print(np.shape(X_train), np.shape(y_train))
print(np.shape(X_train_event), np.shape(y_train_event))


# In[25]:


# Check for the of the training and validation sets
print(np.shape(X_train), np.shape(X_valid))
print(np.shape(X_train_b_jets), np.shape(X_valid_b_jets))
print(np.shape(y_train), np.shape(y_valid))


# In[26]:

# In[27]:

#Define the learning rate schedule for the transformer
class TransformerSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

#Create the callback which clocks the time taken to train
class TimingCallback(keras.callbacks.Callback):
    def __init__(self, logs={}):
        self.logs = []

    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(timer() - self.starttime)


# In[28]:

#Initialises a transformer model.
ParticleTranformer = hffragT.hffragTransformer(
    track_layers=track_layers,
    jet_layers=jet_layers,
    d_model=d_model,
    num_heads=num_heads,
    MASKVAL=MASKVAL,
    dff=dff,
    n_targets=n_targets,
    n_features=6)


# In[29]:

#Creates the optimizer used to train the transformer
learning_rating = TransformerSchedule(d_model)
optimizer = tf.keras.optimizers.Nadam(LR, clipnorm=6)


# In[30]:

#Attach an optimizer and loss to the transformer
ParticleTranformer.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.MSE
)


# In[31]:

#Plots how the learning rate evolves with steps
plt.plot(learning_rating(tf.range(40000, dtype=tf.float32)))
plt.ylabel("Learning_rate")
plt.xlabel("Train Step")


# In[32]:

#Test to see whether the transformer is currently accepting the inputs 
#and producing the desired outputs
output = ParticleTranformer((X_train, X_train_b_jets))
print(output)


# In[33]:


print(ParticleTranformer.summary())


# In[34]:


# Introduce early_stopping to prevent overfitting
early_stopping = callbacks.EarlyStopping(
    min_delta=0.01,  # The minimum amount of change to count as an improvement
    patience=10,  # The number of epochs to wait before stopping
    restore_best_weights=True,  # Keep the best weights
)
# Prevent spikes in the validation and training loss due to the gradient descent kicking the network out of a local minima
reduce_learn_on_plateau = callbacks.ReduceLROnPlateau(
    monitor='loss', factor=0.80, patience=10, min_lr=1e-8)

# Save the weights of the model to allow reuse in future.
path = "/home/physics/phujdj/DeepLearningParticlePhysics/CheckPoints_2/TransformerWeights&Biases.ckpt"
checkpoint_dir = os.path.dirname(path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=path,
                                                 save_weights_only=True, verbose=0, save_freq=100*BATCHSIZE)
# Timer
cb = TimingCallback()

# Weight&Biases Callback:
#Wanda = WandbCallback(save_graph = True,save_weights_only = True, log_weights = True, log_gradients = True, log_evaluation = True, training_data = ((X_train,X_train),y_train), validation_data = ((X_valid,X_valid,),y_valid), log_batch_frequency = 5)


# In[35]:


# Trains the neural network
Training_data = X_train
Validation_data = X_valid
history = ParticleTranformer.fit(
    (X_train, X_train_b_jets), y_train,
    validation_data=((X_valid, X_valid_b_jets), y_valid),
    batch_size=BATCHSIZE,
    epochs=EPOCHS,
    callbacks=[early_stopping, reduce_learn_on_plateau, cb, cp_callback]
)


# In[36]:


# Plot the loss and validation curves vs epoch
history_df = pd.DataFrame(history.history)
fig, ax = plt.subplots()
ax = history_df.loc[:, ['loss', 'val_loss']].plot()
fig.savefig(
    '/home/physics/phujdj/DeepLearningParticlePhysics/TransformerLossCurveProjection.png')
history_df.to_csv(
    '/home/physics/phujdj/DeepLearningParticlePhysics/TransformerhistoryProjection.csv')


# In[37]:


print(sum(cb.logs))


# In[38]:


# Output to the console the minimum epoch
print("Minimum validation loss: {}".format(history_df["loss"].min()))


# In[39]:


# Evaluate the entire performance of the model
loss = ParticleTranformer.evaluate(
    (tracks_scaled, b_jets_scaled), bhads_projection, verbose=2)
print("The Loaded DeepNet has loss: ", loss)

Predictions = ParticleTranformer.predict((tracks_scaled, b_jets_scaled))
Predictions_px = Predictions[:,0]
Predictions_py = Predictions[:,1]
Predictions_pz = Predictions[:,2]
print(Predictions.shape)


Error_px = bhads_projection - Predictions_px
Pull_bhads_px = Error_px/np.std(bhads_projection)

Error_py = bhads_projection[:,1] - Predictions_py
Pull_bhads_py = Error_py/np.std(bhads_projection[:,1])

Error_pz = bhads_projection[:,2] - Predictions_pz
Pull_bhads_pz = Error_pz/np.std(bhads_projection[:,2])

print(np.mean(Error_px))
print(np.std(Error_px))
fig = binneddensity(Error_px, fixedbinning(-60000,60000,100),xlabel = "Error_px")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/Error_PxHffragTransformerProjection.png')
plt.close()

print(np.mean(Error_py))
print(np.std(Error_py))
fig = binneddensity(Error_py, fixedbinning(-60000,60000,100),xlabel = "Error_py")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/Error_PyTransformerProjection.png')
plt.close()

print(np.mean(Error_pz))
print(np.std(Error_pz))
fig = binneddensity(Error_pz, fixedbinning(-60000,60000,100),xlabel = "Error_pz")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/Error_PyHffragTransformerProjection.png')
plt.close()

print(np.mean(Pull_bhads_px))
print(np.std(Pull_bhads_px))
binneddensity(Pull_bhads_px, fixedbinning(-3, 3, 100), xlabel="Pull_px")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/Pull_Bhads_PxHffragTransformerProjection.png')
plt.close()

fig, ax = plt.subplots(figsize=(12, 12))
sns.scatterplot(
    y=Predictions_px,
    x=bhads_projection,
    color="blue"
)
ax.set_title("Scatterplot of the true vs pred X momenta")
ax.set_xlabel("The true X momenta of the tracks from each event")
ax.set_ylabel("The predicted X momenta of b hadron jets")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/ScatterplotPxHffragTransformerProjection.png')
plt.close()
