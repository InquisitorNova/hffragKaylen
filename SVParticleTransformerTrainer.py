#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import RocCurveDisplay
from sklearn import metrics
#from wandb.keras import WandbCallback
from keras import callbacks
import keras
import DeepSetNeuralNetArchitecture as DSNNA
from DeepSetNeuralNetArchitecture import LogNormal_Loss_Function
from DeepSetNeuralNetArchitecture import Mean_Squared_Error
from HffragDeepSetsProjectionMultivariate import DeepSetsProjection
import hffragTransformerJetMultivariateSVFinder as hffragT
from sklearn.feature_selection import mutual_info_regression
import keras.backend as k
import uproot
import awkward as ak
import sklearn as sk
from numpy.lib.recfunctions import structured_to_unstructured
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pandas as pd
from hffrag import fixedbinning
from hffrag import binneddensity
from keras.utils.vis_utils import plot_model
from timeit import default_timer as timer


# In[2]:


plt.style.use("default")
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rc('text',usetex = False)
plt.rc('font',family = 'Times New Roman')


# In[3]:


# The data is being stored in a tree datastructure.
# We access the charm root using this command
tree = uproot.open("/storage/epp2/phswmv/data/hffrag/hffrag.root:CharmAnalysis")


# In[4]:


# Initial parameters
MASKVAL = -999 # This value is introduced to ensure arrays are regular (Of the same size). They will be masked later by the network
MAXTRACKS = 40 # This value is the maximum number of tracks allowed per event
BATCHSIZE = 64 # This is the batch size of the mini batches used during training
EPOCHS = 1  # This is the default number of epochs for which the neural network will train providing that early stopping does not occur
MAXEVENTS = 1e5 #This is the maximum number of events that will the program will accept
LR = 1e-4 #This is the default learning rate


# In[5]:


# Select the features we wish to study
track_features = ["AnalysisTracks_pt", "AnalysisTracks_eta", "AnalysisTracks_phi", "AnalysisTracks_z0sinTheta",
                  "AnalysisTracks_d0sig", "AnalysisTracks_d0", "AnalysisTracks_d0sigPV", "AnalysisTracks_d0PV"]
jet_features = ["AnalysisAntiKt4TruthJets_pt", "AnalysisAntiKt4TruthJets_eta", "AnalysisAntiKt4TruthJets_phi", "AnalysisAntiKt4TruthJets_m","AnalysisAntiKt4TruthJets_ghostB_pdgId",
                "AnalysisAntiKt4TruthJets_ghostB_pt", "AnalysisAntiKt4TruthJets_ghostB_eta","AnalysisAntiKt4TruthJets_ghostB_phi", "AnalysisAntiKt4TruthJets_ghostB_m"]
SV_features = ["TruthParticles_Selected_LxyT"]


# In[6]:


# Read in the data from the root file
features = tree.arrays(jet_features+track_features + SV_features, entry_stop=MAXEVENTS)


# In[7]:


# Select the events of interest
events = features[ak.sum(
    features["AnalysisAntiKt4TruthJets_pt"] > 25000, axis=1) > 0]


# In[8]:


# Displays the number of jets being trained on
jets = events[jet_features][:,0]
print("The number of jets to train on is: ", len(jets))
print("The number of track features is: ",len(track_features))


# In[9]:


# Select tracks from the events
tracks = events[track_features]
Secondary_Displacement_temp_first = events[SV_features]

# Match the tracks to the jets
mask = DSNNA.Match_Tracks(jets, tracks)
matchedtracks = tracks[mask]

# Pad and Flatten the data
matchedtracks = DSNNA.flatten(matchedtracks, MAXTRACKS)


# In[10]:


bjets = ak.sum(jets["AnalysisAntiKt4TruthJets_ghostB_pt"] > 5000, axis=1) > 0
Secondary_Displacement_temp_a = DSNNA.flatten(Secondary_Displacement_temp_first, 6).to_numpy()
Secondary_Displacement_temp_b = Secondary_Displacement_temp_a[:,0]
Secondary_Displacement_temp = Secondary_Displacement_temp_b[bjets]
jets = jets[bjets]

# Obtain the pt, eta and phi of each b hadron jet
bhads_pt = jets["AnalysisAntiKt4TruthJets_ghostB_pt"][:, 0].to_numpy()
bhads_eta = jets["AnalysisAntiKt4TruthJets_ghostB_eta"][:,0].to_numpy()
bhads_phi = jets["AnalysisAntiKt4TruthJets_ghostB_phi"][:,0].to_numpy()
bhads_m = jets["AnalysisAntiKt4TruthJets_ghostB_m"][:,0].to_numpy()
bhads_PGID = jets["AnalysisAntiKt4TruthJets_ghostB_pdgId"][:,0].to_numpy()

jets_pt = jets["AnalysisAntiKt4TruthJets_pt"].to_numpy()
jets_eta = jets["AnalysisAntiKt4TruthJets_eta"].to_numpy()
jets_phi = jets["AnalysisAntiKt4TruthJets_phi"].to_numpy()
jets_m = jets["AnalysisAntiKt4TruthJets_m"].to_numpy()
b_jets = np.stack([jets_pt,jets_eta,jets_phi, jets_m], axis = -1)

bhads = np.stack([bhads_pt,bhads_eta,bhads_phi, bhads_m],axis = -1) #Combine the momentum, eta and phi for each jet into one array

print("There are {} outputs".format(np.shape(bhads)[1])) # Display the number of target features the neural network will predict
matchedtracks = matchedtracks[bjets]
print("There are {} inputs".format(np.shape(matchedtracks)[1])) # Display the number of target features the neural network will use in it's predictions


# In[11]:


Secondary_Displacement = np.array([x[0] for x in Secondary_Displacement_temp])
print(np.min(Secondary_Displacement), np.max(Secondary_Displacement))


# In[12]:


# Transform the jet and tracks to unstructed data.
jets = structured_to_unstructured(jets[jet_features[:-5]])
matchedtracks = structured_to_unstructured(matchedtracks)


# In[13]:


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


# In[14]:


tracks = np.ma.masked_values(tracks,-999)


# In[15]:


bhads_fractions_px = bhads[:,0]/b_jets[:,0]
bhads_fractions_py = bhads[:,1]/b_jets[:,1]
bhads_fractions_pz = bhads[:,2]/b_jets[:,2]
print(bhads_fractions_px.shape)


# In[16]:


b_jets_mag = np.linalg.norm(b_jets[:,:3], axis = 1)
bhads_fractions = np.stack([bhads_fractions_px,bhads_fractions_py, bhads_fractions_pz], axis = -1)
bhads_projection = ((bhads[:,:3]*b_jets[:,:3]).sum(axis = 1))/(b_jets_mag**2)
print(bhads_fractions.shape)


# In[17]:


print(np.max(bhads_fractions), np.min(bhads_fractions_px))
array = [x for x in range(bhads_fractions_px.shape[0])]
bhads_trial = np.stack([array,bhads_fractions_px, bhads_fractions_py, bhads_fractions_pz, bhads_projection, Secondary_Displacement], axis = -1)
bhads_fractions_clean  = bhads_trial[(bhads_trial[:,1] < 1.5) & (bhads_trial[:,1] > 0) & (bhads_trial[:,2] < 1.5) & (bhads_trial[:,2] > 0) & (bhads_trial[:,3] < 1.5) & (bhads_trial[:,3] > 0) & (bhads_trial[:,4] > 0) & (bhads_trial[:,4] <1.5) & (bhads_trial[:,5] > 0) * (bhads_trial[:,5] < 20)]
print(bhads_fractions_clean.shape)
print(np.max(bhads_fractions_clean[:,1]), np.min(bhads_fractions_clean[:,1]))
indices = bhads_fractions_clean[:,0]
print(indices.shape)


# In[18]:


indices = [int(x) for x in indices]
print(np.shape(indices))


# In[19]:


tracks = tracks[indices]
b_jets = b_jets[indices]
bhads = bhads[indices]
bhads_pt = bhads_pt[indices]
bhads_eta = bhads_eta[indices]
bhads_phi = bhads_phi[indices]
bhads_PGID = bhads_PGID[indices]
b_jets_m = b_jets_m[indices]
bhads_m = bhads_m[indices]
Secondary_Displacement = Secondary_Displacement[indices]


# In[20]:


Tracks_Momentum = np.sqrt(tracks[:, : ,0]**2 + tracks[:,:,1]**2 + tracks[:,:,2]**2)
Tracks_4_Momentum = np.stack([Tracks_Momentum, tracks[:,:,0], tracks[:,:,1], tracks[:,:,2]], axis = -1)
print(Tracks_4_Momentum.shape)


# In[21]:


Tracks_Invariant_Mass = np.sqrt((np.sum(Tracks_4_Momentum, axis = 1) * np.sum(Tracks_4_Momentum, axis = 1)).sum(axis = -1))
print(Tracks_Invariant_Mass.shape)


# In[22]:


np.min(Secondary_Displacement), np.max(Secondary_Displacement), np.quantile(Secondary_Displacement, 0.75)


# In[23]:


b_jets_mag = np.linalg.norm(b_jets[:,:3], axis = 1)
bhads_mag = np.linalg.norm(bhads[:,:3], axis = 1)
tracks_Momentum = np.sum(np.linalg.norm(tracks[:,:,:3], axis = 2))

bhads_fractions_px = bhads[:,0]/b_jets[:,0]
bhads_fractions_py = bhads[:,1]/b_jets[:,1]
bhads_fractions_pz = bhads[:,2]/b_jets[:,2]
print(bhads_fractions_px.shape)

b_jets_energy = np.sqrt((b_jets_m[:,0]**2) + (b_jets_mag**2))
print(b_jets_energy.shape, b_jets_m.shape)

b_jets_energy_pt = np.sqrt((b_jets_m[:,0]**2) + (b_jets[:,4]**2))
b_jets_energy_pt.shape

b_jet_energy_mass_ratio = b_jets_energy/b_jets_m[:,0]

print(b_jet_energy_mass_ratio.shape)
bhads_energy = np.sqrt((bhads_m**2) + (bhads_mag**2))
bhads_energy.shape

bhads_energy_mass_ratio = bhads_energy/bhads_m

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
bhads_transverse_mass = np.sqrt(bhads_energy**2 - bhads[:,2]**2)
print(b_jet_transverse_mass[0])
print(b_jet_energy_ratio_total.shape)
print(b_jet_transverse_mass.shape)
print(np.full((len(b_jets)),b_jet_energy_ratio_total).shape)
print("end")
bhads_projection = ((bhads[:,:3]*b_jets[:,:3]).sum(axis = 1))/(b_jets_mag**2)

print(np.mean(b_jets_energy),np.std(b_jets_energy))

b_jets = np.stack([b_jets[:,0], b_jets[:,1], b_jets[:,2],b_jets[:,3],b_jets[:,4], b_jets[:,5], b_jets[:,6], b_jets_mag, sum_px_tracks, sum_py_tracks, sum_pz_tracks, sum_pt_tracks, sum_px_tracks_RSE, sum_py_tracks_RSE, sum_pz_tracks_RSE, sum_pt_tracks_RSE, RSM_scaled_px, RSM_scaled_py, RSM_scaled_pz, RSM_scaled_pt, RMS_scaled_px, RMS_scaled_py, RMS_scaled_pz, RMS_scaled_pt, b_jet_transverse_mass, Log_Sum_px, Log_Sum_py, Log_Sum_pz, Log_Sum_pt, Log_Momenta, b_jets_energy, b_jet_energy_ratio_px, b_jet_energy_ratio_py, b_jet_energy_ratio_pz, b_jet_energy_ratio_cart, b_jet_energy_ratio_pt, b_jet_energy_mass_ratio, np.full((len(b_jets),),b_jet_energy_ratio_total)], axis = -1)
bhads_targets = np.stack([bhads[:,0],bhads[:,1], bhads[:,2], bhads_pt, bhads_eta, bhads_phi, bhads_fractions_px, bhads_fractions_py, bhads_fractions_pz, bhads_energy, bhads_transverse_mass, bhads_energy_mass_ratio, bhads_projection, bhads_m, Secondary_Displacement], axis = -1)


# In[24]:


"""
0 - Momentum Px Scaled
1 - Momentum Py Scaled
2 - Momentum Pz Scaled
3-  Momentum Pt Scaled
4-  Momentum Eta Scaled
5 - Momentum Phi Scaled
6 - Tranvserse Impact parameter sin component
7 - Longitudinal Impact parameter
8 - Longitudinal Impact parameter signficiance
9 - Longitudinal Impact Parameter w.r.t PV
10 - Longitudinal Impact Parameter wr.r.t PV
11 - Momentum Fraction Px Scaled
12 - Momentum Fraction Py Scaled
13 - Momentum Fraction Pz Scaled
14 - Momentum Fraction pt Scaled.
15 - Logarithm of px of the tracks / b_jet x momenta
16 - Logarithm of py of the tracks / b_jet y momenta
17 - Logarithm of pz of the tracks / b_jet z momenta
18 - Logarithm of sum pt of the tracks / b_jet t momenta
"""
Tracks_input = np.concatenate([tracks, Track_fractions, Log_tracks], axis = -1)
print(Tracks_input.shape)


# In[25]:


"""
0 - Momentum Px Scaled
1 - Momentum Py Scaled
2 - Momentum Pz Scaled
3 - Momentum Pt Scaled
4 - Momentum eta Scaled
5 - Momentum Phi Scaled
6 - Sum px of the tracks
7 - Sum py of the tracks
8 - Sum pz of the tracks
9 - Sum pt of the tracks
10 - Sqrt of the Sum px of the tracks
11 - Sqrt of the Sum py of the tracks
12 - Sqrt of the Sum pz of the tracks
13 - Sqrt of the Sum pt of the tracks
14 - Sqrt of the Sum px of the tracks scaled by the sum px of the tracks
15 - Sqrt of the Sum py of the tracks scaled by the sum py of the tracks
16 - Sqrt of the Sum pz of the tracks scaled by the sum pz of the tracks
17 - Sqrt of the Sum pt of the tracks scaked by the sum pt of the tracks
18 - Root Mean Square of the px momenta of the tracks
19 - Root Mean Square of the py momenta of the tracks
20 - Root Mean Square of the pz momenta of the tracks
21 - Root Mean Square of the pt momenta of the tracks
22 - Tranvserse mass of the b-jets
23 - Logarithm of the  Sum px of the tracks divide by the b_jet x momenta
24 - Logarithm of the  Sum py of the tracks divide by the b_jet y momenta
25 - Logarithm of the  Sum pz of the tracks divide by the b_jet z momenta
26 - Logarithm of the total momenta of the tracks divided by the b_jet total momenta
27 - B_jet energy
28 - B_jet energy ratio px
29 - B_jet energy ratio py
30 - B_jet energy ratio pz
31 - B_jet energy ratio pt
32 - B_jet energy ratio cart
32 - B_jet energy ratio pt
33 - B_jet energy ratio total
"""
b_jets_input = np.concatenate([b_jets, Tracks_projection, Sum_Tracks_projection.reshape(-1,1)], axis = -1)
print(b_jets_input.shape)


# In[26]:


from collections import Counter
counter = Counter(bhads_PGID)
print(counter)

from sklearn.preprocessing import OneHotEncoder
counter = np.array([])
bhadron_PGIDs = []
for PGID in bhads_PGID:
    if (PGID == 521 or PGID == -521):
        element = 0
    elif (PGID == 511 or PGID == -511):
        element = 1
    elif (PGID == 531 or PGID == -531):
        element = 2
    else:
        element = 4
    counter = np.append(counter, [element])
    bhadron_PGIDs.append([str(element)])
print(np.shape(bhadron_PGIDs))
counter = Counter(counter)
print(counter)
encoder = OneHotEncoder(sparse = False)
onehot = encoder.fit_transform(bhadron_PGIDs)
print(np.shape(onehot))


# In[27]:


mask = np.where(np.isinf(b_jets_input) == True)
b_jets_input_clean = np.delete(b_jets_input, mask, axis = 0)
print(b_jets_input_clean.shape)
Tracks_input_clean = np.delete(Tracks_input, mask, axis = 0)
bhads_m_clean = np.delete(bhads_m, mask, axis = 0)
bhads_targets_clean = np.delete(bhads_targets, mask, axis = 0)
onehot = np.delete(onehot, mask, axis = 0)  
print(Tracks_input_clean.shape,b_jets_input_clean.shape,bhads_m_clean.shape, bhads_targets_clean.shape)


# In[28]:


Scaler_tracks = StandardScaler()
Num_events,Num_tracks,Num_features = np.shape(Tracks_input_clean)
Scaled_tracks = np.reshape(Tracks_input_clean, newshape=(-1,Num_features))
tracks_scaled = Scaler_tracks.fit_transform(Scaled_tracks)
Tracks_input_scaled = np.reshape(tracks_scaled, newshape= (Num_events,Num_tracks,Num_features))
print(np.shape(Tracks_input_scaled))

Scaler_jets = StandardScaler()
Num_events,Num_features = np.shape(b_jets_input_clean)
b_jets_scaled = np.reshape(b_jets_input_clean, newshape=(-1,Num_features))
b_jets_scaled = Scaler_jets.fit_transform(b_jets_scaled)
b_jets_input_scaled = np.reshape(b_jets_scaled, newshape= (Num_events,Num_features))
print(np.shape(b_jets_input_scaled))

means = []
stds = []
lister = []
for bhads_target_feature in range(np.shape(bhads_targets_clean)[1]):
    Bhads_targets = bhads_targets_clean[:,bhads_target_feature]
    mean, std = np.mean(Bhads_targets), np.std(Bhads_targets)
    means = np.append(means,mean)
    stds = np.append(stds,std)
    Standardized_Bhads_targets = (Bhads_targets - mean)/(std)
    Standardized_Bhads_targets = Standardized_Bhads_targets.reshape(-1,1)
    lister.append(Standardized_Bhads_targets)
Standardized_Bhads_targets = np.concatenate(lister,axis = 1)
print(Standardized_Bhads_targets.shape)
print(means,stds)


# In[29]:


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

class TimingCallback(keras.callbacks.Callback):
    def __init__(self, logs = {}):
        self.logs = []
    def on_epoch_begin(self, epoch, logs ={}):
        self.starttime = timer()
    def on_epoch_end(self, epoch, logs = {}):
        self.logs.append(timer() - self.starttime)


# In[30]:


num_layers = 3
d_model = 64
dff = 64
num_heads = 4
n_features = np.shape(b_jets_input_scaled)[1]
n_targets = np.shape(Standardized_Bhads_targets)[1]
n_targets_classification = np.shape(onehot)[1]
dropout_rate = 0.01
track_layers = 4
jet_layers = [256,256,256,256]


# In[31]:


ParticleTranformer = hffragT.hffragTransformer(
    track_layers = track_layers,
    n_targets_classifications = n_targets_classification,
    jet_layers =  jet_layers, 
    d_model = d_model, 
    num_heads = num_heads,
    MASKVAL= MASKVAL,
    dff = dff,
    n_targets= n_targets,
    n_features=n_features)


# In[32]:


learning_rating = TransformerSchedule(d_model)
optimizer = tf.keras.optimizers.Nadam(LR)


# In[40]:


from DeepSetNeuralNetArchitecture import LogNormal_Loss_Function
from HffragDeepSetsProjection import Multivariate_Gaussian_Negative_Likelihood_Loss_Curve
ParticleTranformer.compile(
 optimizer = optimizer,
 loss = {"output_1":Multivariate_Gaussian_Negative_Likelihood_Loss_Curve, "output_2":tf.keras.losses.categorical_crossentropy},
 metrics = [Mean_Squared_Error]   
)


# In[34]:


output = ParticleTranformer((Tracks_input_scaled,b_jets_input_scaled))
output


# In[35]:


ParticleTranformer.summary()


# In[36]:


# Introduce early_stopping to prevent overfitting
early_stopping = callbacks.EarlyStopping(
    min_delta=0.00001,  # The minimum amount of change to count as an improvement
    patience=50,  # The number of epochs to wait before stopping
    restore_best_weights=True,  # Keep the best weights
)
# Prevent spikes in the validation and training loss due to the gradient descent kicking the network out of a local minima
reduce_learn_on_plateau = callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.80, patience=15, min_lr=1e-9)

# Save the weights of the model to allow reuse in future.
path = "/home/physics/phujdj/DeepLearningParticlePhysics/CheckPointsDeepNet/TransformerWeights&Biases.ckpt"
checkpoint_dir = os.path.dirname(path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=path,
                                                 save_weights_only=True, verbose=0, save_freq = 100*BATCHSIZE)
#Timer
#cb = TimingCallback()

#Weight&Biases Callback:
#Wanda = WandbCallback(save_graph = True,save_weights_only = True, log_weights = True, log_gradients = True, log_evaluation = True, training_data = (X_train,y_train), validation_data = (X_valid,y_valid), log_batch_frequency = 5)

# Learning Scheduler:
exponential_decay_fn = DSNNA.expontial_decay(lr0 = LR,s = 100)
learning_scheduler = tf.keras.callbacks.LearningRateScheduler(exponential_decay_fn)


# In[41]:


history  = ParticleTranformer.fit(
    (Tracks_input_scaled,b_jets_input_scaled), y = {"output_1":Standardized_Bhads_targets, "output_2":onehot},
    validation_split = 0.3,
    epochs = EPOCHS,
    batch_size = BATCHSIZE,
    callbacks = [early_stopping, cp_callback, reduce_learn_on_plateau],
    )



# Plot the loss and validation curves vs epoch
history_df = pd.DataFrame(history.history)
fig,ax = plt.subplots()
ax = history_df.loc[:, ['loss', 'val_loss']].plot()
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/TransformerLossCurve.png')
history_df.to_csv('/home/physics/phujdj/DeepLearningParticlePhysics/Transformerhistory.csv')


# Output to the console the minimum epoch
print("Minimum validation loss: {}".format(history_df["loss"].min()))

#Evaluate the entire performance of the model
loss = ParticleTranformer.evaluate((Tracks_input_scaled,b_jets_input_scaled),(Standardized_Bhads_targets, onehot),verbose = 2)
print("The Loaded DeepNet has loss: ", loss)

predictions = ParticleTranformer.predict((Tracks_input_scaled, b_jets_input_scaled))

Predictions = predictions[0]
Predictions_Mass = predictions[1]

lister = []
index = 0
predictions_x = Predictions[:,:np.shape(Standardized_Bhads_targets)[1]]
for bhads_features in range(np.shape(predictions_x)[1]):
    Predictions_temp = predictions_x[:, bhads_features]
    Predictions_temp = (Predictions_temp*stds[index]) + means[index]
    Predictions_temp = Predictions_temp.reshape(-1,1)
    lister.append(Predictions_temp)
    index+=1
Predictions_X = np.concatenate(lister, axis = 1)
print(Predictions_X)

lister = []
index = 0
predictions_x_uncertainty = Predictions[:,np.shape(Standardized_Bhads_targets)[1]:2*np.shape(Standardized_Bhads_targets)[1]]
print(predictions_x_uncertainty)
for bhads_features in range(np.shape(predictions_x_uncertainty)[1]):
    Predictions_temp = predictions_x_uncertainty[:, bhads_features]
    Predictions_temp = (Predictions_temp*stds[index]) + means[index]
    Predictions_temp = Predictions_temp.reshape(-1,1)
    lister.append(Predictions_temp)
    index+=1
Predictions_Uncertainty = np.concatenate(lister, axis = 1)
print(Predictions_Uncertainty)

Error_px_unscaled = bhads_targets_clean[:,0] - Predictions_X[:,0]
Pull_bhads_px_unscaled = Error_px_unscaled/np.std(bhads_targets_clean[:,0])
Pull_px = Error_px_unscaled/Predictions_Uncertainty[:,0]

Error_py_unscaled = bhads_targets_clean[:,1] - Predictions_X[:,1]
Pull_bhads_py_unscaled = Error_py_unscaled/np.std(bhads_targets_clean[:,1])
Pull_py = Error_py_unscaled/Predictions_Uncertainty[:,1]

Error_pz_unscaled = bhads_targets_clean[:,2] - Predictions_X[:,2]
Pull_bhads_pz_unscaled = Error_pz_unscaled/np.std(bhads_targets_clean[:,2])
Pull_pz = Error_pz_unscaled/Predictions_Uncertainty[:,2]

Error_pt_unscaled = bhads_targets_clean[:,3] - Predictions_X[:,3]
Pull_bhads_pt_unscaled = Error_pt_unscaled/np.std(bhads_targets_clean[:,3])
Pull_pt = Error_pt_unscaled/Predictions_Uncertainty[:,3]

Error_eta_unscaled = bhads_targets_clean[:,4] - Predictions_X[:,4]
Pull_bhads_eta_unscaled = Error_eta_unscaled/np.std(bhads_targets_clean[:,4])
Pull_eta = Error_eta_unscaled/Predictions_Uncertainty[:,4]

Error_phi_unscaled = bhads_targets_clean[:,5] - Predictions_X[:,5]
Pull_bhads_pt_unscaled = Error_pt_unscaled/np.std(bhads_targets_clean[:,5])
Pull_phi = Error_phi_unscaled/Predictions_Uncertainty[:,5]

Error_pxfraction_unscaled = bhads_targets_clean[:,6] - Predictions_X[:,6]
Pull_pxfraction_unscaled = Error_pxfraction_unscaled/np.std(bhads_targets_clean[:,6])
Pull_pxfraction = Error_pxfraction_unscaled/Predictions_Uncertainty[:,6]

Error_pyfraction_unscaled = bhads_targets_clean[:,7] - Predictions_X[:,7]
Pull_pyfraction_unscaled = Error_pyfraction_unscaled/np.std(bhads_targets_clean[:,7])
Pull_pyfraction = Error_pyfraction_unscaled/Predictions_Uncertainty[:,7]

Error_pzfraction_unscaled = bhads_targets_clean[:,8] - Predictions_X[:,8]
Pull_pzfraction_unscaled = Error_pzfraction_unscaled/np.std(bhads_targets_clean[:,8])
Pull_pzfraction = Error_pzfraction_unscaled/Predictions_Uncertainty[:,8]

Error_energy_unscaled = bhads_targets_clean[:,9] - Predictions_X[:,9]
Pull_energy_unscaled = Error_energy_unscaled/np.std(bhads_targets_clean[:,9])
Pull_energy = Error_energy_unscaled/Predictions_Uncertainty[:,8]

Error_projection_unscaled = bhads_targets_clean[:,12] - Predictions_X[:,12]
Pull_bhads_projection_unscaled = Error_projection_unscaled/np.std(bhads_targets_clean[:,12])
Pull_projection = Error_projection_unscaled/Predictions_Uncertainty[:,12]

Error_mass_unscaled = bhads_targets_clean[:,13] - Predictions_X[:, 13]
Pull_mass_unscaled = Error_energy_unscaled/np.std(bhads_targets_clean[:,13])
Pull_mass = Error_energy_unscaled/Predictions_Uncertainty[:,13]

Error_Secondary_Displacement_unscaled = bhads_targets_clean[:,14] - Predictions_X[:, 14]
Pull_Secondary_Displacement_unscaled = Error_Secondary_Displacement_unscaled/np.std(bhads_targets_clean[:,14])
Pull_Secondary_Displacement = Error_Secondary_Displacement_unscaled/Predictions_Uncertainty[:,14]

print(np.mean(Error_px_unscaled))
print(np.std(Error_px_unscaled))
fig = binneddensity(Error_px_unscaled, fixedbinning(-20000,20000,100),xlabel = "Error_px_unscaled")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/Error_PxHffragTransformer.png')
plt.close()

print(np.mean(Pull_px))
print(np.std(Pull_px))
fig = binneddensity(Pull_px, fixedbinning(-2,2,100),xlabel = "Pull_Px")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/Pull_PxHffragTransformer.png')
plt.close()

print(np.mean(Error_py_unscaled))
print(np.std(Error_py_unscaled))
fig = binneddensity(Error_py_unscaled, fixedbinning(-20000,20000,100),xlabel = "Error_py_unscaled")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/Error_PyHffragTransformer.png')
plt.close()

print(np.mean(Pull_py))
print(np.std(Pull_py))
fig = binneddensity(Pull_py, fixedbinning(-2,2,100),xlabel = "Pull_Py")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/Pull_PyHffragTransformer.png')
plt.close()

print(np.mean(Error_pz_unscaled))
print(np.std(Error_pz_unscaled))
fig = binneddensity(Error_pz_unscaled, fixedbinning(-40000,40000,100),xlabel = "Error_pz_unscaled")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/Error_PzHffragTransformer.png')
plt.close()

print(np.mean(Pull_pz))
print(np.std(Pull_pz))
fig = binneddensity(Pull_pz, fixedbinning(-2,2,100),xlabel = "Pull_Pz")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/Pull_PzHffragTransformer.png')
plt.close()

print(np.mean(Error_projection_unscaled))
print(np.std(Error_projection_unscaled))
fig = binneddensity(Error_projection_unscaled, fixedbinning(-1,1,100),xlabel = "Error_projection")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/Error_Projection_HffragTransformer.png')
plt.close()

print(np.mean(Pull_projection))
print(np.std(Pull_projection))
fig = binneddensity(Pull_projection, fixedbinning(-1,1,100),xlabel = "Pull_projection")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/Pull_Projection_HffragTransformer.png')
plt.close()

print(np.mean(Error_pxfraction_unscaled))
print(np.std(Error_pxfraction_unscaled))
fig = binneddensity(Error_pxfraction_unscaled, fixedbinning(-1,1,100),xlabel = "Error_px_fraction")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/Error_px_fraction_HffragTransformer.png')
plt.close()

print(np.mean(Pull_pxfraction_unscaled))
print(np.std(Pull_pxfraction_unscaled))
fig = binneddensity(Pull_pxfraction_unscaled, fixedbinning(-2,2,100),xlabel = "Pull_px_fraction")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/Pull_px_fraction_HffragTransformer.png')
plt.close()

print(np.mean(Error_pyfraction_unscaled))
print(np.std(Error_pyfraction_unscaled))
fig = binneddensity(Error_pyfraction_unscaled, fixedbinning(-1,1,100),xlabel = "Error_py_fraction")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/Error_py_fraction_HffragTransformer.png')
plt.close()

print(np.mean(Pull_pyfraction_unscaled))
print(np.std(Pull_pyfraction_unscaled))
fig = binneddensity(Pull_pyfraction_unscaled, fixedbinning(-2,2,100),xlabel = "Pull_py_fraction")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/Pull_py_fraction_HffragTransformer.png')
plt.close()

print(np.mean(Error_pzfraction_unscaled))
print(np.std(Error_pzfraction_unscaled))
fig = binneddensity(Error_pzfraction_unscaled, fixedbinning(-1,1,100),xlabel = "Error_pz_fraction")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/Error_pz_fraction_HffragTransformer.png')
plt.close()

print(np.mean(Pull_pzfraction_unscaled))
print(np.std(Pull_pzfraction_unscaled))
fig = binneddensity(Pull_pzfraction_unscaled, fixedbinning(-2,2,100),xlabel = "Pull_pz_fraction")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/Pull_pz_fraction_HffragTransformer.png')
plt.close()

print(np.mean(Error_energy_unscaled))
print(np.std(Error_energy_unscaled))
fig = binneddensity(Error_energy_unscaled, fixedbinning(-40000,40000,100),xlabel = "Error_energy")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/Error_energy_HffragTransformer.png')
plt.close()

print(np.mean(Pull_energy_unscaled))
print(np.std(Pull_energy_unscaled))
fig = binneddensity(Pull_energy_unscaled, fixedbinning(-1,1,100),xlabel = "Pull_energy_HffragTransformer.png")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/Pull_energy_HffragTransformer.png')
plt.close()

print(np.mean(Error_mass_unscaled))
print(np.std(Error_mass_unscaled))
fig = binneddensity(Error_mass_unscaled, fixedbinning(-250,500,100),xlabel = "Mass Error")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/Mass_Error_HffragTransformer.png')
plt.close()

print(np.mean(Error_pt_unscaled))
print(np.std(Error_pt_unscaled))
fig = binneddensity(Error_pt_unscaled, fixedbinning(-40000,40000,100),xlabel = "Error_pt")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/Error_pt_HffragTransformer.png')
plt.close()

print(np.mean(Pull_pt))
print(np.std(Pull_pt))
fig = binneddensity(Pull_pt, fixedbinning(-1,1,100),xlabel = "Pull_pt")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/Pull_pt_HffragTransformer.png')
plt.close()

print(np.mean(Error_eta_unscaled))
print(np.std(Error_eta_unscaled))
fig = binneddensity(Error_eta_unscaled, fixedbinning(-0.1,0.1,100),xlabel = "Error_eta")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/Error_eta_HffragTransformer.png')
plt.close()

print(np.mean(Pull_eta))
print(np.std(Pull_eta))
fig = binneddensity(Pull_eta, fixedbinning(-3,3,100),xlabel = "Pull_eta")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/Pull_eta_HffragTransformer.png')
plt.close()

print(np.mean(Error_phi_unscaled))
print(np.std(Error_phi_unscaled))
fig = binneddensity(Error_phi_unscaled, fixedbinning(-0.1,0.1,100),xlabel = "Error_phi")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/Error_phi_HffragTransformer.png')
plt.close()

print(np.mean(Pull_phi))
print(np.std(Pull_phi))
fig = binneddensity(Pull_phi, fixedbinning(-3,3,100),xlabel = "Pull_phi")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/Pull_phi_HffragTransformer.png')
plt.close()

print(np.mean(Error_Secondary_Displacement_unscaled))
print(np.std(Error_Secondary_Displacement_unscaled))
fig = binneddensity(Error_Secondary_Displacement_unscaled, fixedbinning(-10,10,100),xlabel = "Error_SecondaryVertex")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/Error_Secondary_Displacements_HffragTransformer.png')
plt.close()

print(np.mean(Pull_Secondary_Displacement))
print(np.std(Pull_Secondary_Displacement))
fig = binneddensity(Pull_Secondary_Displacement, fixedbinning(-1,1,100),xlabel = "Pull_SecondaryVertex")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/Pull_Secondary_Displacements_HffragTransformer.png')
plt.close()

fig, ax = plt.subplots(figsize = (12,12))
x = bhads_targets_clean[:,0]
y = Predictions_X[:,0]
grid = np.vstack([x, y])
z = gaussian_kde(grid)(grid)
idx = z.argsort()
x,y,z = x[idx], y[idx], z[idx]

im = sns.scatterplot(
    x,
    y,
    c = z,
    s = 5,
    cmap = "autumn",
    )
norm = plt.Normalize(z.min(), z.max())
sm = plt.cm.ScalarMappable(cmap = "autumn", norm = norm)
sm.set_array([])
im.figure.colorbar(sm)
ax.set_title("Scatterplot of the true vs pred X momenta")
ax.set_xlabel("The true X momenta of the tracks from each event")
ax.set_ylabel("The predicted X momenta of b hadron jets")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/PxScatter_HffragTransformer.png')
plt.close()

fig, ax = plt.subplots(figsize = (12,12))
x = bhads_targets_clean[:,1]
y = Predictions_X[:,1]
grid = np.vstack([x, y])
z = gaussian_kde(grid)(grid)
idx = z.argsort()
x,y,z = x[idx], y[idx], z[idx]

im = sns.scatterplot(
    x,
    y,
    c = z,
    s = 5,
    cmap = "inferno",
    )
norm = plt.Normalize(z.min(), z.max())
sm = plt.cm.ScalarMappable(cmap = "inferno", norm = norm)
sm.set_array([])
im.figure.colorbar(sm)
ax.set_title("Scatterplot of the true vs pred Y momenta")
ax.set_xlabel("The true Y momenta of the b-hadrons from each event")
ax.set_ylabel("The predicted Y momenta of b hadron jets")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/PyScatter_HffragTransformer.png')
plt.close()

fig, ax = plt.subplots(figsize = (12,12))
x = bhads_targets_clean[:,2]
y = Predictions_X[:,2]
grid = np.vstack([x, y])
z = gaussian_kde(grid)(grid)
idx = z.argsort()
x,y,z = x[idx], y[idx], z[idx]

im = sns.scatterplot(
    x,
    y,
    c = z,
    s = 5,
    cmap = "winter",
    )
norm = plt.Normalize(z.min(), z.max())
sm = plt.cm.ScalarMappable(cmap = "winter", norm = norm)
sm.set_array([])
im.figure.colorbar(sm)
ax.set_title("Scatterplot of the true vs pred Z momenta")
ax.set_xlabel("The true Z momenta of the b-hadrons from each event")
ax.set_ylabel("The predicted Z momenta of b hadron jets")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/PzScatter_HffragTransformer.png')
plt.close()

fig, ax = plt.subplots(figsize = (12,12))
x = bhads_targets_clean[:,3]
y = Predictions_X[:,3]
grid = np.vstack([x, y])
z = gaussian_kde(grid)(grid)
idx = z.argsort()
x,y,z = x[idx], y[idx], z[idx]

im = sns.scatterplot(
    x,
    y,
    c = z,
    s = 5,
    cmap = "spring",
    )
norm = plt.Normalize(z.min(), z.max())
sm = plt.cm.ScalarMappable(cmap = "spring", norm = norm)
sm.set_array([])
im.figure.colorbar(sm)
ax.set_title("Scatterplot of the true vs pred transverse momenta")
ax.set_xlabel("The true tranvserse momenta of the b-hadrons from each event")
ax.set_ylabel("The predicted transverse momenta of b hadron jets")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/PtScatter_HffragTransformer.png')
plt.close()

fig, ax = plt.subplots(figsize = (12,12))
x = bhads_targets_clean[:,4]
y = Predictions_X[:,4]
grid = np.vstack([x, y])
z = gaussian_kde(grid)(grid)
idx = z.argsort()
x,y,z = x[idx], y[idx], z[idx]

im = sns.scatterplot(
    x,
    y,
    c = z,
    s = 5,
    cmap = "gist_heat",
    )
norm = plt.Normalize(z.min(), z.max())
sm = plt.cm.ScalarMappable(cmap = "gist_heat", norm = norm)
sm.set_array([])
im.figure.colorbar(sm)
ax.set_title("Scatterplot of the true vs pred pseudorapidity")
ax.set_xlabel("The true pseudorapidity of the b-hadrons from each event")
ax.set_ylabel("The predicted pseudorapidity of b hadron jets")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/EtaScatter_HffragTransformer.png')
plt.close()

fig, ax = plt.subplots(figsize = (12,12))
x = bhads_targets_clean[:,5]
y = Predictions_X[:,5]
grid = np.vstack([x, y])
z = gaussian_kde(grid)(grid)
idx = z.argsort()
x,y,z = x[idx], y[idx], z[idx]

im = sns.scatterplot(
    x,
    y,
    c = z,
    s = 5,
    cmap = "summer",
    )
norm = plt.Normalize(z.min(), z.max())
sm = plt.cm.ScalarMappable(cmap = "summer", norm = norm)
sm.set_array([])
im.figure.colorbar(sm)
ax.set_title("Scatterplot of the true vs pred phi")
ax.set_xlabel("The true phi of the b-hadrons from each event")
ax.set_ylabel("The predicted phi of b hadron jets")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/PhiScatter_HffragTransformer.png')
plt.close()

fig, ax = plt.subplots(figsize = (12,12))
x = bhads_targets_clean[:,6]
y = Predictions_X[:,6]
grid = np.vstack([x, y])
z = gaussian_kde(grid)(grid)
idx = z.argsort()
x,y,z = x[idx], y[idx], z[idx]

im = sns.scatterplot(
    x,
    y,
    c = z,
    s = 5,
    cmap = "spring",
    )
norm = plt.Normalize(z.min(), z.max())
sm = plt.cm.ScalarMappable(cmap = "spring", norm = norm)
sm.set_array([])
im.figure.colorbar(sm)
ax.set_title("Scatterplot of the true vs pred X momenta fraction")
ax.set_xlabel("The true X momenta fraction of the b-hadrons from each event")
ax.set_ylabel("The predicted X fraction of b hadron jets")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/Px_fraction_HffragTransformer.png')
plt.close()

fig, ax = plt.subplots(figsize = (12,12))
x = bhads_targets_clean[:,7]
y = Predictions_X[:,7]
grid = np.vstack([x, y])
z = gaussian_kde(grid)(grid)
idx = z.argsort()
x,y,z = x[idx], y[idx], z[idx]

im = sns.scatterplot(
    x,
    y,
    c = z,
    s = 5,
    cmap = "gist_heat",
    )
norm = plt.Normalize(z.min(), z.max())
sm = plt.cm.ScalarMappable(cmap = "gist_heat", norm = norm)
sm.set_array([])
im.figure.colorbar(sm)
ax.set_title("Scatterplot of the true vs pred Y momenta fraction")
ax.set_xlabel("The true Y momenta fraction of the b-hadrons from each event")
ax.set_ylabel("The predicted Y fraction of b hadron jets")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/Py_fraction_HffragTransformer.png')
plt.close()

fig, ax = plt.subplots(figsize = (12,12))
x = bhads_targets_clean[:,8]
y = Predictions_X[:,8]
grid = np.vstack([x, y])
z = gaussian_kde(grid)(grid)
idx = z.argsort()
x,y,z = x[idx], y[idx], z[idx]

im = sns.scatterplot(
    x,
    y,
    c = z,
    s = 5,
    cmap = "summer",
    )
norm = plt.Normalize(z.min(), z.max())
sm = plt.cm.ScalarMappable(cmap = "summer", norm = norm)
sm.set_array([])
im.figure.colorbar(sm)
ax.set_title("Scatterplot of the true vs pred Z momenta fraction")
ax.set_xlabel("The true Z momenta fraction of the b-hadrons from each event")
ax.set_ylabel("The predicted Z fraction of b hadron jets")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/Pz_fraction_HffragTransformer.png')
plt.close()

fig, ax = plt.subplots(figsize = (12,12))
x = bhads_targets_clean[:,9]
y = Predictions_X[:,9]
grid = np.vstack([x, y])
z = gaussian_kde(grid)(grid)
idx = z.argsort()
x,y,z = x[idx], y[idx], z[idx]

im = sns.scatterplot(
    x,
    y,
    c = z,
    s = 5,
    cmap = "PiYG",
    )
norm = plt.Normalize(z.min(), z.max())
sm = plt.cm.ScalarMappable(cmap = "PiYG", norm = norm)
sm.set_array([])
im.figure.colorbar(sm)
ax.set_title("Scatterplot of the true vs pred Bhadron energy")
ax.set_xlabel("The true Bhadron energy of the b-hadrons from each event")
ax.set_ylabel("The predicted Bhadron energy of b hadron jets")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/Bhad_Energy_HffragTransformer.png')
plt.close()

fig, ax = plt.subplots(figsize = (12,12))
x = bhads_targets_clean[:,12]
y = Predictions_X[:,12]
grid = np.vstack([x, y])
z = gaussian_kde(grid)(grid)
idx = z.argsort()
x,y,z = x[idx], y[idx], z[idx]

im = sns.scatterplot(
    x,
    y,
    c = z,
    s = 5,
    cmap = "PiYG",
    )
norm = plt.Normalize(z.min(), z.max())
sm = plt.cm.ScalarMappable(cmap = "PiYG", norm = norm)
sm.set_array([])
im.figure.colorbar(sm)
ax.set_title("Scatterplot of the true vs pred Bhadron projection")
ax.set_xlabel("The true Bhadron projection of the b-hadrons from each event")
ax.set_ylabel("The predicted Bhadron projection of b hadron jets")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/Bhad_Projection_HffragTransformer.png')
plt.close()

fig, ax = plt.subplots(figsize = (12,12))
x = bhads_targets_clean[:,13]
y = Predictions_X[:,13]
grid = np.vstack([x, y])
z = gaussian_kde(grid)(grid)
idx = z.argsort()
x,y,z = x[idx], y[idx], z[idx]

im = sns.scatterplot(
    x,
    y,
    c = z,
    s = 5,
    cmap = "PiYG",
    )
norm = plt.Normalize(z.min(), z.max())
sm = plt.cm.ScalarMappable(cmap = "PiYG", norm = norm)
sm.set_array([])
im.figure.colorbar(sm)
ax.set_title("Scatterplot of the true vs pred Bhadron mass")
ax.set_xlabel("The true Bhadron mass of the b-hadrons from each event")
ax.set_ylabel("The predicted Bhadron mass of b hadron jets")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/Bhad_Mass_HffragTransformer.png')
plt.close()

fig, ax = plt.subplots(figsize = (12,12))
x = bhads_targets_clean[:,14]
y = Predictions_X[:,14]
grid = np.vstack([x, y])
z = gaussian_kde(grid)(grid)
idx = z.argsort()
x,y,z = x[idx], y[idx], z[idx]

im = sns.scatterplot(
    x,
    y,
    c = z,
    s = 5,
    cmap = "PiYG",
    )
norm = plt.Normalize(z.min(), z.max())
sm = plt.cm.ScalarMappable(cmap = "PiYG", norm = norm)
sm.set_array([])
im.figure.colorbar(sm)
ax.set_title("Scatterplot of the true vs pred Bhadron Secondary Vertex Displacement")
ax.set_xlabel("The true Bhadron Secondary Vertex Displacement of the b-hadrons from each event")
ax.set_ylabel("The predicted Bhadron Secondary Vertex Displacement of b hadron jets")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/Secondary_Displacement_HffragTransformer.png')
plt.close()


print(classification_report(np.argmax(onehot, axis = 1), np.argmax(Predictions_Mass,axis = 1)))

confusion_mat = confusion_matrix(np.argmax(onehot, axis = 1), np.argmax(Predictions_Mass,axis = 1))
confusion_visualized = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_mat)
fig,ax = plt.subplots()
ax = confusion_visualized.plot()
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/ConfusionMatrix.png')
plt.close()


target_names = ["Mass_0", "Mass_1", "Mass_2", "Mass_3", "Mass_4", "Mass_5"]
fig, ax = plt.subplots(figsize = (6,6))
colors = cycle(["aqua", "darkorange", "cornflowerblue"])
for class_id, colors in zip(range(4), colors):
    RocCurveDisplay.from_predictions(
        onehot[:,class_id],
        Predictions_Mass[:, class_id],
        name = f"ROC curve for {target_names[class_id]}",
        color = colors,
        ax = ax,
    )
plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("One-vs-Rest ROC curves:\n Bhadron Masses vs the Rest:")
plt.legend()
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/ROC.png')
plt.close()
