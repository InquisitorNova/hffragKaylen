#!/usr/bin/env python
# coding: utf-8

# In[1]:
"""
*Filename: hffragTransformerTrainer
*Description: In this jupyter notebook the hffragTransformerTransformer is trained
*using the gaussian negative loss likelihood function. Once trained the program returns
*the resolutions plots and scatterplots of the true vs predicted.
Date: 16/02/2023
Author: Kaylen Smith Darnbrook
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from wandb.keras import WandbCallback
from keras import callbacks
import keras
import keras.backend as k
import uproot
import awkward as ak
import sklearn as sk
from numpy.lib.recfunctions import structured_to_unstructured
from sklearn.model_selection import train_test_split
import tensorflow as tf
from hffragTransformerArchitecture import hffragTransformer
from hffragTransformerArchitecture import LogNormal_Loss_Function
import DeepSetNeuralNetArchitecture as DSNNA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
import pandas as pd
from hffrag import fixedbinning
from hffrag import binneddensity
from keras.utils.vis_utils import plot_model
from timeit import default_timer as timer
import matplotlib.pyplot as plt

# In[2]:


#wandb.init(project = "hffrag-TransformerNeuralNetworkArchitecture")


# In[2]:


# The data is being stored in a tree datastructure.
# We access the charm root using this command
tree = uproot.open("hffrag.root:CharmAnalysis")


# In[3]:


# Initial parameters
MASKVAL = -999 # This value is introduced to ensure arrays are regular (Of the same size). They will be masked later by the network
MAXTRACKS = 32 # This value is the maximum number of tracks allowed per event
BATCHSIZE = 64 # This is the batch size of the mini batches used during training
EPOCHS = 2000 # This is the default number of epochs for which the neural network will train providing that early stopping does not occur
MAXEVENTS = 1e20 #This is the maximum number of events that will the program will accept
LR = 1e-4 #This is the default learning rate


# In[4]:

#Initialise the hyperparameters for the transformer
track_layers = [64,64,64,64]
num_layers = 6
num_heads = 8
dff = 512
MASKVAL = -999
dropout_rate = 0.005
n_targets = 3
d_model = MAXTRACKS


# In[6]:


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
print("There are {} inputs".format(np.shape(matchedtracks)[1])) # Display the number of target features the neural network will use in it's ppredictions


# In[11]:


print(np.shape(bhads)) #Check the shape of the neural network
print(np.shape(jet_features[:-1])) #Check for shape of the jet features
print(jets[jet_features[0]]) # Check the jets


# In[13]:
jets_pt = jets["AnalysisAntiKt4TruthJets_pt"].to_numpy()
jets_eta = jets["AnalysisAntiKt4TruthJets_eta"].to_numpy()
jets_phi = jets["AnalysisAntiKt4TruthJets_phi"].to_numpy()
b_jets = np.stack([jets_pt,jets_eta,jets_phi], axis = -1)


# Transform the jet and tracks to unstructed data.
jets = structured_to_unstructured(jets[jet_features[:-3]])
matchedtracks = structured_to_unstructured(matchedtracks)


# In[14]:


#Check the matchtracks are the correct shape
print(matchedtracks[:, 0:1])
print(np.shape(matchedtracks[:, :, 3]))


# In[15]:


# Convert the coordinates of the b jets and tracks to cartesian coordinates
tracks_p = DSNNA.pt_eta_phi_2_px_py_pz_tracks(matchedtracks.to_numpy())
bhads = DSNNA.pt_eta_phi_2_px_py_pz_jets(bhads)
b_jets = DSNNA.pt_eta_phi_2_px_py_pz_jets(b_jets)

#Combine the momenta of the tracks with the rest of the track features to form the track dataset
tracks = np.concatenate([tracks_p,matchedtracks[:,:,3:].to_numpy()],axis = 2)

Scaler = StandardScaler()
Num_events,Num_tracks,Num_features = np.shape(tracks)
tracks = np.reshape(tracks, newshape=(-1,Num_features))
tracks = Scaler.fit_transform(tracks)
tracks = np.reshape(tracks, newshape= (Num_events,Num_tracks,Num_features))

bhads_scaled = bhads/b_jets
print(bhads_scaled[0])

# In[16]:


#Check that this is all the correct shape
print(np.shape(tracks))
print(np.shape(bhads))
print(tracks[0,0])
print(bhads[0])


# In[17]:


# Split the data into training and validation sets.
X_train, X_valid, y_train, y_valid = train_test_split(
    tracks, bhads_scaled, train_size=0.8, random_state=42)


# In[18]:

#Single Event Test Cases.
X_train_event, y_train_event = np.array([X_train[0]]), np.array([y_train[0]])
X_valid_event, y_valid_event = np.array([X_valid[0]]), np.array([y_valid[0]])
print(np.shape(X_train),np.shape(y_train))
print(np.shape(X_train_event),np.shape(y_train_event))


# In[19]:


#Check for the of the training and validation sets
print(np.shape(X_train), np.shape(X_valid))
print(np.shape(y_train), np.shape(y_valid))


# In[23]:


# In[20]:

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
    def __init__(self, logs = {}):
        self.logs = []
    def on_epoch_begin(self, epoch, logs ={}):
        self.starttime = timer()
    def on_epoch_end(self, epoch, logs = {}):
        self.logs.append(timer() - self.starttime)


# In[21]:

#Initialises a transformer model.
sample_hffragTransformer = hffragTransformer(
    track_layers=track_layers,
    num_layers = num_layers,
    dff = dff,
    num_heads=num_heads,
    MASKVAL = MASKVAL,
    dropout= dropout_rate,
    n_targets= n_targets,
    d_model = d_model
)


# In[22]:

#Test to see whether the transformer is currently accepting the inputs 
#and producing the desired outputs
output = sample_hffragTransformer(X_train)
print(sample_hffragTransformer.summary())


# In[23]:

#Creates the optimizer used to train the transformer
learning_rating = TransformerSchedule(d_model)
optimizer = tf.keras.optimizers.Nadam(learning_rating,beta_1=0.9, beta_2=0.98, clipnorm = 1.0)


# In[24]:

#Attach an optimizer and loss to the transformer
sample_hffragTransformer.compile(
    optimizer=optimizer,
    loss = LogNormal_Loss_Function
)


# In[30]:



# In[25]:


# Introduce early_stopping to prevent overfitting
early_stopping = callbacks.EarlyStopping(
    min_delta=0.00001,  # The minimum amount of change to count as an improvement
    patience=45,  # The number of epochs to wait before stopping
    restore_best_weights=True,  # Keep the best weights
)
# Prevent spikes in the validation and training loss due to the gradient descent kicking the network out of a local minima
reduce_learn_on_plateau = callbacks.ReduceLROnPlateau(
    monitor='loss', factor=0.95, patience=15, min_lr=1e-8)

# Save the weights of the model to allow reuse in future.
path = "/home/physics/phujdj/DeepLearningParticlePhysics/CheckPointshffragTransformer/hffragTransformerWeights&Biases.ckpt"
checkpoint_dir = os.path.dirname(path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=path,
                                                 save_weights_only=True, verbose=0, save_freq = 100*BATCHSIZE)
#Timer
cb = TimingCallback()

#Weight&Biases Callback:
#Wanda = WandbCallback(save_graph = True,save_weights_only = True, log_weights = True, log_gradients = True, log_evaluation = True, training_data = (X_train,y_train), validation_data = (X_valid,y_valid), log_batch_frequency = 5)


# In[26]:

#Determine the shapes of X_train and y_train
print(np.shape(X_train),np.shape(y_train),np.shape(X_valid),np.shape(y_valid))
print(np.max(output))


# In[34]:

# Trains the neural network
history = sample_hffragTransformer.fit(
    X_train, y_train,
    validation_data= (X_valid, y_valid),
    batch_size=BATCHSIZE,
    epochs=EPOCHS,
    callbacks = [early_stopping,reduce_learn_on_plateau,cb,cp_callback],
    use_multiprocessing=True
)


# In[ ]:


# Plot the loss and validation curves vs epoch
history_df = pd.DataFrame(history.history)
np.log(history_df.loc[:, ["loss","val_loss"]]).plot()
history_df.to_csv('/home/physics/phujdj/DeepLearningParticlePhysics/hffraghistory.csv')


# In[ ]:


print(sum(cb.logs))


# In[ ]:


# Output to the console the minimum epoch
print("Minimum validation loss: {}".format(history_df["loss"].min()))


# In[ ]:


#Evaluate the entire performance of the model
loss = sample_hffragTransformer.evaluate((tracks,tracks),bhads,verbose = 2)
print("The Transformer has loss: ", loss)


# In[ ]:

#Evaluate the performance of the model using resolution, pulls and scatterplots
PredictionsNeural = sample_hffragTransformer.predict(tracks) * b_jets
print(PredictionsNeural.shape)


# In[ ]:


ErrorPx = PredictionsNeural[:,0] - bhads[:,0]
Pull_Px = ErrorPx/PredictionsNeural[:,3]


# In[ ]:


fig = binneddensity(PredictionsNeural[:,0], fixedbinning(-100000,100000,100), xlabel ="Predicted Bhad X Momentum [MeV]")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/TransformerPredictions.png')


# In[ ]:


fig = binneddensity(ErrorPx, fixedbinning(-100000,100000,100), xlabel ="Predicted Bhad X Momentum Error [MeV]")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/TransformerError.png')


# In[ ]:


fig = binneddensity(Pull_Px, fixedbinning(-1,1,100), xlabel ="Predicted Bhad X Momentum Pull")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/TransformerPredictions.png')


# In[ ]:


fig,ax = plt.subplots(figsize = (12,12))
sns.scatterplot(
    x = bhads,
    y = PredictionsNeural,
    color = "purple"
)
ax.set_title("Scatterplot of the true vs pred X momenta")
ax.set_xlim([np.min(PredictionsNeural),np.max(PredictionsNeural)])
ax.set_ylim([np.min(bhads),np.max(bhads)])
ax.set_xlabel("The true X momenta of the tracks from each event")
ax.set_ylabel("The predicted X momenta of b hadron jets")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/TransformerPredictionsScatterplot.png')

