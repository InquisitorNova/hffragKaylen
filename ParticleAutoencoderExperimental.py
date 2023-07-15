#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import seaborn as sns
from keras import callbacks
import keras
from keras import regularizers
import DeepSetNeuralNetArchitecture as DSNNA
from DeepSetNeuralNetArchitecture import LogNormal_Loss_Function
from DeepSetNeuralNetArchitecture import Mean_Squared_Error
from HffragDeepSetsProjectionMultivariate import DeepSetsProjection
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
import matplotlib.figure as figure
from keras.utils.vis_utils import plot_model
from timeit import default_timer as timer
import matplotlib.pyplot as plt

# In[2]:
# returns a fixed set of bin edges
def fixedbinning(xmin, xmax, nbins):
  return np.mgrid[xmin:xmax:nbins*1j]


# define two functions to aid in plotting
def hist(xs, binning, normalized=False):
  ys = np.histogram(xs, bins=binning)[0]

  yerrs = np.sqrt(ys)

  if normalized:
    s = np.sum(ys)
    ys = ys / s
    yerrs = yerrs / s

  return ys, yerrs


def binneddensity(xs, binning, label=None, xlabel=None, ylabel="binned probability density"):
  fig = figure.Figure(figsize=(8, 8))
  plt = fig.add_subplot(111)

  ys , yerrs = hist(xs, binning, normalized=True)

  # determine the central value of each histogram bin
  # as well as the width of each bin
  # this assumes a fixed bin size.
  xs = (binning[1:]+binning[:-1]) / 2.0
  xerrs = ((binning[1:]-binning[:-1]) / 2.0)

  plt.errorbar \
    ( xs
    , ys
    , xerr=xerrs
    , yerr=yerrs
    , label=label
    , linewidth=0
    , elinewidth=2
    )

  plt.set_xlabel(xlabel)
  plt.set_ylabel(ylabel)

  return fig


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
MAXTRACKS = 32 # This value is the maximum number of tracks allowed per event
BATCHSIZE = 64 # This is the batch size of the mini batches used during training
EPOCHS = 1000 # This is the default number of epochs for which the neural network will train providing that early stopping does not occur
MAXEVENTS = 1e6 #This is the maximum number of events that will the program will accept
LR = 5e-4 #This is the default learning rate


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
jets = events[jet_features][:, 0]
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
Secondary_Displacement = np.ma.masked_values(Secondary_Displacement,-999)


# In[15]:


bhads_fractions_px = bhads[:,0]/b_jets[:,0]
bhads_fractions_py = bhads[:,1]/b_jets[:,1]
bhads_fractions_pz = bhads[:,2]/b_jets[:,2]
bhads_fractions_pt = bhads_pt/b_jets_pep[:,0]
print(bhads_fractions_px.shape)

b_jets_mag = np.linalg.norm(b_jets[:,:3], axis = 1)
bhads_fractions = np.stack([bhads_fractions_px,bhads_fractions_py, bhads_fractions_pz], axis = -1)
bhads_projection = ((bhads[:,:3]*b_jets[:,:3]).sum(axis = 1))/(b_jets_mag**2)
print(bhads_fractions_px.shape)



# In[17]:


print(bhads_fractions_px.shape)
print(Secondary_Displacement.shape)


# In[166]:


#Identify the outliers in the dataset and remove them to prevent spikes during learning.
print(np.max(bhads_fractions_px), np.min(bhads_fractions_px))
print(np.max(bhads_fractions_py), np.min(bhads_fractions_py))
print(np.max(bhads_fractions_pz), np.min(bhads_fractions_pz))
print(np.min(bhads_fractions_pt), np.min(bhads_fractions_pt))
array = [x for x in range(bhads_fractions_px.shape[0])]

#Remove the outliers in the dataset with momenta fractions and projections that are either unphysical or statistically unlikely
bhads_trial = np.stack([array,bhads_fractions_px, bhads_fractions_py, bhads_fractions_pz, bhads_projection, Secondary_Displacement], axis = -1)
bhads_fractions_clean  = bhads_trial[(bhads_trial[:,1] <= 10) & (bhads_trial[:,1] >= -10) & (bhads_trial[:,2] <= 10) & (bhads_trial[:,2] >= -10) & (bhads_trial[:,3] <= 10) & (bhads_trial[:,3] >= -10) & (bhads_trial[:,4] >= -10) & (bhads_trial[:,4] <= 10) & (bhads_trial[:,5] >= -10) * (bhads_trial[:,5] <= 10)]
print(bhads_fractions_clean.shape)

#Compare to the original range of the momenta fractions
print(np.max(bhads_fractions_clean[:,1]), np.min(bhads_fractions_clean[:,1]))
indices = bhads_fractions_clean[:,0]
print(indices.shape)
print(len(indices)/len(bhads))


#Find the indices of the remaining datapoints
indices = [int(x) for x in indices]
print(np.shape(indices))


# In[142]:


tracks_original = tracks
b_jets_original = b_jets
bhads_original = bhads
bhads_pt_original = bhads_pt
bhads_eta_original = bhads_eta
bhads_phi_original = bhads_phi
bhads_PGID_original = bhads_PGID
b_jets_pep_original = b_jets_pep
b_jets_m_original = b_jets_m
bhads_m_original = bhads_m
Secondary_Displacement_original = Secondary_Displacement


# In[163]:

# In[143]:


#Removes the outliers from the data
tracks = tracks[indices]
b_jets = b_jets[indices]
bhads = bhads[indices]
bhads_pt = bhads_pt[indices]
bhads_eta = bhads_eta[indices]
bhads_phi = bhads_phi[indices]
bhads_PGID = bhads_PGID[indices]
b_jets_pep = b_jets_pep[indices]
b_jets_m = b_jets_m[indices]
bhads_m = bhads_m[indices]
bhads_fractions_pt = bhads_fractions_pt[indices]
Secondary_Displacement = Secondary_Displacement[indices]

# In[16]:


Tracks_Momentum = np.sqrt(tracks[:, : ,0]**2 + tracks[:,:,1]**2 + tracks[:,:,2]**2)
Tracks_4_Momentum = np.stack([Tracks_Momentum, tracks[:,:,0], tracks[:,:,1], tracks[:,:,2]], axis = -1)
print(Tracks_4_Momentum.shape)
Tracks_Invariant_Mass = np.sqrt((np.sum(Tracks_4_Momentum, axis = 1) * np.sum(Tracks_4_Momentum, axis = 1)).sum(axis = -1))
print(Tracks_Invariant_Mass.shape)


# In[17]:


Secondary_Displacement = np.ma.masked_values(Secondary_Displacement,-999)


# In[18]:


b_jets_mag = np.linalg.norm(b_jets[:,:3], axis = 1)
bhads_mag = np.linalg.norm(bhads[:,:3], axis = 1)
tracks_Momentum = np.sum(np.linalg.norm(tracks[:,:,:3], axis = 2))

bhads_fractions_px = bhads[:,0]/b_jets[:,0]
bhads_fractions_py = bhads[:,1]/b_jets[:,1]
bhads_fractions_pz = bhads[:,2]/b_jets[:,2]
bhads_fractions_pt = bhads_pt/b_jets_pep[:,0]
print(bhads_fractions_px.shape)
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
bhads_targets = np.stack([bhads[:,0]/1e6,bhads[:,1]/1e6, bhads[:,2]/1e6,bhads_pt/1e6, bhads_eta, bhads_phi, bhads_fractions_px, bhads_fractions_py, bhads_fractions_pz, bhads_energy/1e6, bhads_fractions_pt, bhads_transverse_mass/1e6, bhads_energy_mass_ratio/1e3, bhads_projection, bhads_m/1e3, Secondary_Displacement/1e4], axis = -1)


# In[19]:


bhads_targets = np.ma.masked_values(bhads_targets, -999)
np.min(bhads_projection), np.max(bhads_projection)


# In[20]:


"""
0 - Momentum Px 
1 - Momentum Py
2 - Momentum Pz
3 - Momentum Pt
4 - Total Momentum
5-  Momentum Eta Scaled
6 - Momentum Phi Scaled
7 - Tranvserse Impact parameter sin component
8 - Longitudinal Impact parameter
9 - Longitudinal Impact parameter signficiance
10 - Longitudinal Impact Parameter w.r.t PV
11 - Longitudinal Impact Parameter wr.r.t PV
12 - Momentum Fraction Px Scaled
13 - Momentum Fraction Py Scaled
14 - Momentum Fraction Pz Scaled
16 - Momentum Fraction pt Scaled.
17 - Logarithm of px of the tracks / b_jet x momenta
18 - Logarithm of py of the tracks / b_jet y momenta
19 - Logarithm of pz of the tracks / b_jet z momenta
20 - Logarithm of sum pt of the tracks / b_jet t momenta
"""
Tracks_input = np.concatenate([tracks, Tracks_4_Momentum[:,:,0].reshape(-1,32,1), Track_fractions, Log_tracks], axis = -1)
print(Tracks_input.shape)


# In[21]:


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
b_jets_input = np.concatenate([b_jets, Tracks_projection, Sum_Tracks_projection.reshape(-1,1), Tracks_Invariant_Mass.reshape(-1,1)], axis = -1)
print(b_jets_input.shape)


# In[22]:


mask = np.where(np.isinf(b_jets_input) == True)
b_jets_input_clean = np.delete(b_jets_input, mask, axis = 0)
print(b_jets_input_clean.shape)
Tracks_input_clean = np.delete(Tracks_input, mask, axis = 0)
bhads_m_clean = np.delete(bhads_m, mask, axis = 0)
bhads_targets_clean = np.delete(bhads_targets, mask, axis = 0) 
print(Tracks_input_clean.shape,b_jets_input_clean.shape,bhads_m_clean.shape, bhads_targets_clean.shape)


# In[23]:


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


# In[24]:


def TrackEncoder(Num_Tracks_Features, bottleneck, MASKVAL = -999):

    Track_features = keras.layers.Input(shape = [None, Num_Tracks_Features])

    Track_layers = keras.layers.Masking(MASKVAL)(Track_features)

    #Encoder:
    Track_layers_Embedding = keras.layers.TimeDistributed(keras.layers.Dense(Num_Tracks_Features*2, activation = "gelu", kernel_initializer = "he_normal", kernel_regularizer = regularizers.l1_l2(1e-4)))(Track_layers)
    Track_layers_Embedding = keras.layers.BatchNormalization()(Track_layers_Embedding)
    Track_layers_Embedding = keras.layers.TimeDistributed(keras.layers.Dense(round(Num_Tracks_Features)/2, activation = "gelu", kernel_initializer = "he_normal",  kernel_regularizer = regularizers.l1_l2(1e-4)))(Track_layers_Embedding)
    Track_layers_Embedding = keras.layers.BatchNormalization()(Track_layers_Embedding)
    Track_layers_Embedding = keras.layers.TimeDistributed(keras.layers.Dense(round(Num_Tracks_Features)/4, activation = "gelu", kernel_initializer = "he_normal",  kernel_regularizer = regularizers.l1_l2(1e-4)))(Track_layers_Embedding)
    Track_layers_Embedding = keras.layers.BatchNormalization()(Track_layers_Embedding)
    Track_layers_Embedding = keras.layers.TimeDistributed(keras.layers.Dense(bottleneck, activation = "gelu", kernel_initializer = "he_normal",  kernel_regularizer = regularizers.l1_l2(1e-4)))(Track_layers_Embedding)
    Track_layers_Embedding = keras.layers.BatchNormalization()(Track_layers_Embedding)
    
    #Decoder:
    Track_layers_Reconstructed = keras.layers.TimeDistributed(keras.layers.Dense(bottleneck, activation = "gelu", kernel_initializer = "he_normal",  kernel_regularizer = regularizers.l1_l2(1e-4)))(Track_layers_Embedding)
    Track_layers_Reconstructed = keras.layers.BatchNormalization()(Track_layers_Reconstructed)
    Track_layers_Reconstructed = keras.layers.TimeDistributed(keras.layers.Dense(Num_Tracks_Features/4, activation = "gelu", kernel_initializer = "he_normal",  kernel_regularizer = regularizers.l1_l2(1e-4)))(Track_layers_Reconstructed)
    Track_layers_Reconstructed = keras.layers.BatchNormalization()(Track_layers_Reconstructed)
    Track_layers_Reconstructed = keras.layers.TimeDistributed(keras.layers.Dense(Num_Tracks_Features/2, activation = "gelu", kernel_initializer = "he_normal",  kernel_regularizer = regularizers.l1_l2(1e-4)))(Track_layers_Reconstructed)
    Track_layers_Reconstructed = keras.layers.BatchNormalization()(Track_layers_Reconstructed)
    Track_layers_Reconstructed = keras.layers.TimeDistributed(keras.layers.Dense(Num_Tracks_Features, activation = "gelu", kernel_initializer = "he_normal",  kernel_regularizer = regularizers.l1_l2(1e-4)))(Track_layers_Reconstructed)

    #Output
    output = keras.layers.TimeDistributed(keras.layers.Dense(Num_Tracks_Features, activation = "linear"))(Track_layers_Reconstructed)

    Model = keras.Model(inputs = Track_features, outputs = output)

    return Model


# In[25]:


def JetEncoder(Num_Jets_Features, bottleneck, MASKVAL = -999):

    Jet_features = keras.layers.Input(shape = [Num_Jets_Features])

    Jet_layers = keras.layers.Masking(MASKVAL)(Jet_features)

    #Encoder:
    Jet_layers_Embedding = keras.layers.Dense(Num_Jets_Features*2, activation = "gelu", kernel_initializer = "he_normal")(Jet_layers)
    Jet_layers_Embedding = keras.layers.BatchNormalization()(Jet_layers_Embedding)
    Jet_layers_Embedding = keras.layers.Dense(Num_Jets_Features, activation = "gelu", kernel_initializer = "he_normal")(Jet_layers)
    Jet_layers_Embedding = keras.layers.BatchNormalization()(Jet_layers_Embedding)
    Jet_layers_Embedding = keras.layers.Dense(round(Num_Jets_Features/2), activation = "gelu", kernel_initializer = "he_normal")(Jet_layers_Embedding)
    Jet_layers_Embedding = keras.layers.BatchNormalization()(Jet_layers_Embedding)
    Jet_layers_Embedding = keras.layers.Dense(round(Num_Jets_Features/4), activation = "gelu", kernel_initializer = "he_normal")(Jet_layers_Embedding)
    Jet_layers_Embedding = keras.layers.BatchNormalization()(Jet_layers_Embedding)
    Jet_layers_Embedding = keras.layers.Dense(round(Num_Jets_Features/8), activation = "gelu", kernel_initializer = "he_normal")(Jet_layers_Embedding)
    Jet_layers_Embedding = keras.layers.BatchNormalization()(Jet_layers_Embedding)
    Jet_layers_Embedding = keras.layers.Dense(bottleneck, activation = "gelu", kernel_initializer = "he_normal")(Jet_layers_Embedding)
    Jet_layers_Embedding = keras.layers.BatchNormalization()(Jet_layers_Embedding)
    
    #Decoder:
    Jet_layers_Reconstructed = keras.layers.Dense(bottleneck, activation = "gelu", kernel_initializer = "he_normal")(Jet_layers_Embedding)
    Jet_layers_Reconstructed = keras.layers.BatchNormalization()(Jet_layers_Reconstructed)
    Jet_layers_Reconstructed = keras.layers.Dense(round(Num_Jets_Features/8), activation = "gelu", kernel_initializer = "he_normal")(Jet_layers_Reconstructed)
    Jet_layers_Reconstructed = keras.layers.BatchNormalization()(Jet_layers_Reconstructed)
    Jet_layers_Reconstructed = keras.layers.Dense(round(Num_Jets_Features/4), activation = "gelu", kernel_initializer = "he_normal")(Jet_layers_Reconstructed)
    Jet_layers_Reconstructed = keras.layers.BatchNormalization()(Jet_layers_Reconstructed)
    Jet_layers_Reconstructed = keras.layers.Dense(round(Num_Jets_Features/2), activation = "gelu", kernel_initializer = "he_normal")(Jet_layers_Reconstructed)
    Jet_layers_Reconstructed = keras.layers.BatchNormalization()(Jet_layers_Reconstructed)
    Jet_layers_Reconstructed = keras.layers.Dense(Num_Jets_Features, activation = "gelu", kernel_initializer = "he_normal")(Jet_layers_Reconstructed)

    #Output
    output = keras.layers.Dense(Num_Jets_Features, activation = "linear")(Jet_layers_Reconstructed)

    Model = keras.Model(inputs = Jet_features, outputs = output)

    return Model


# In[26]:


ParticleEncoder = TrackEncoder(np.shape(Tracks_input_clean)[2], 5)


# In[27]:


ParticleEncoder.compile(
    loss = keras.losses.huber,
    optimizer = tf.keras.optimizers.Nadam(LR, clipnorm = 1.0, use_ema = True),
)


# In[28]:


print(ParticleEncoder.summary())


# In[29]:


# In[39]:


# Split the data into training and validation sets.
X_train, X_valid, y_train, y_valid = train_test_split(
    Tracks_input_scaled, Standardized_Bhads_targets, train_size=0.7, random_state = 42)


# In[40]:


# Split the data into training and validation sets.
X_train_b_jets, X_valid_b_jets, y_train_b_jets, y_valid_b_jets = train_test_split(
    b_jets_input_scaled, Standardized_Bhads_targets, train_size=0.7, random_state = 42)


# In[32]:


# Introduce early_stopping to prevent overfitting
early_stopping = callbacks.EarlyStopping(
    min_delta=0.001,  # The minimum amount of change to count as an improvement
    patience=30,  # The number of epochs to wait before stopping
    restore_best_weights=True,  # Keep the best weights
)
# Prevent spikes in the validation and training loss due to the gradient descent kicking the network out of a local minima
reduce_learn_on_plateau = callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.80, patience=15, min_lr=1e-6)

# Save the weights of the model to allow reuse in future.
path = "/home/physics/phujdj/DeepLearningParticlePhysics/CheckPointsDeepNet/TrackAutoEncoderWeights&Biases.ckpt"
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


history_1 = ParticleEncoder.fit(
    X_train, X_train,
    validation_data= (X_valid, X_valid),
    epochs= 300,
    batch_size= BATCHSIZE,
     callbacks = [early_stopping, cp_callback, reduce_learn_on_plateau],
    )


# In[34]:


# Plot the loss and validation curves vs epoch
history_df = pd.DataFrame(history_1.history)
fig,ax = plt.subplots()
ax = (history_df.loc[:, ['loss', 'val_loss']]).plot()
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/TrackAutoEncoderLoss.png')
history_df.to_csv('/home/physics/phujdj/DeepLearningParticlePhysics/TrackAutoEncoderLoss.csv')

# In[42]:


#Evaluate the entire performance of the model
loss = ParticleEncoder.evaluate(Tracks_input_scaled,Tracks_input_scaled,verbose = 2)
print("The TrackParticleEncoder has loss: ", loss)


# In[43]:


BParticleEncoder = JetEncoder(np.shape(b_jets_input_scaled)[1], 9)


# In[44]:


BParticleEncoder.compile(
    loss = keras.losses.huber,
    optimizer = tf.keras.optimizers.Nadam(LR, use_ema = True),
)


# In[45]:


print(BParticleEncoder.summary())


# In[ ]:

# In[46]:


history_2 = BParticleEncoder.fit(
    X_train_b_jets, X_train_b_jets,
    validation_data= (X_valid_b_jets, X_valid_b_jets),
    epochs= 300,
    batch_size= BATCHSIZE,
     callbacks = [early_stopping, cp_callback, reduce_learn_on_plateau],
    )


# In[47]:


# Plot the loss and validation curves vs epoch
history_df_1 = pd.DataFrame(history_2.history)
fig,ax = plt.subplots()
ax = (history_df_1.loc[:, ['loss', 'val_loss']]).plot()
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/JetAutoEncoderLoss.png')
history_df_1.to_csv('/home/physics/phujdj/DeepLearningParticlePhysics/JetAutoEncoderLoss.csv')

# In[50]:


#Evaluate the entire performance of the model
loss = BParticleEncoder.evaluate(b_jets_input_scaled,b_jets_input_scaled,verbose = 2)
print("The JetParticleEncoder has loss: ", loss)


# In[48]:


ParticleEncoderTracksInput = keras.layers.Input(ParticleEncoder.layers[0].input_shape[0][1:])
ParticleEncoderTracks = ParticleEncoderTracksInput
for layer in ParticleEncoder.layers[1:9]:
    print(layer)
    ParticleEncoderTracks = layer(ParticleEncoderTracks)
ParticleEncoderTracks = keras.Model(inputs = ParticleEncoderTracksInput, outputs = ParticleEncoderTracks)
ParticleEncoderTracks.save_weights("/home/physics/phujdj/DeepLearningParticlePhysics/CheckPointsDeepNet/TrackEncoderModelWeights&Biases.ckpt")

# In[49]:


ParticleEncoderJetsInput = keras.layers.Input(BParticleEncoder.layers[0].input_shape[0][1:])
ParticleEncoderJets = ParticleEncoderJetsInput
for layer in BParticleEncoder.layers[1:9]:
    print(layer)
    ParticleEncoderJets = layer(ParticleEncoderJets)
ParticleEncoderJets = keras.Model(inputs = ParticleEncoderJetsInput, outputs = ParticleEncoderJets)
ParticleEncoderJets.save_weights("/home/physics/phujdj/DeepLearningParticlePhysics/CheckPointsDeepNet/JetEncoderModelWeights&Biases.ckpt")

# In[51]:


print(ParticleEncoderTracks.summary())


# In[52]:


print(ParticleEncoderJets.summary())


# In[53]:


ParticleEncoderTracksClone = keras.models.clone_model(ParticleEncoderTracks)
ParticleEncoderTracksClone.set_weights(ParticleEncoderTracks.get_weights())


# In[54]:


ParticleEncoderJetsClone = keras.models.clone_model(ParticleEncoderJets)
ParticleEncoderJetsClone.set_weights(ParticleEncoderJets.get_weights())


# In[55]:


X_train_encoded = ParticleEncoderTracksClone.predict(X_train) 
X_valid_encoded = ParticleEncoderTracksClone.predict(X_valid)


# In[56]:


tracks_encoded = ParticleEncoderTracksClone.predict(Tracks_input_scaled)


# In[57]:


X_train_b_jets_encoded = ParticleEncoderJetsClone.predict(X_train_b_jets)
X_valid_b_jets_encoded = ParticleEncoderJetsClone.predict(X_valid_b_jets)


# In[58]:


jets_encoded = ParticleEncoderJetsClone.predict(b_jets_input_scaled)


# In[59]:


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
onehot = np.delete(onehot, mask, axis = 0)  
print(np.shape(onehot))


# In[60]:


# Builds the deep neural network
track_layers = [20,30,40]
jet_layers = [250 for x in range(3)]
b_jets_layers = [20,30,40]

track_layers = [np.shape(tracks_encoded)[2]]+track_layers
print(track_layers)

#Initializers the optimizer used for training the network
optimizer = tf.keras.optimizers.Nadam(LR, use_ema=True)

#Build a DeepSets Projection Neural Network
DeepSetProjector = DeepSetsProjection(track_layers=track_layers, jet_layers=jet_layers, b_jet_layers=b_jets_layers, n_targets_classification=np.shape(onehot)[1], regularizer_strength = 1e-4, n_features = np.shape(jets_encoded)[1], n_targets=np.shape(bhads_targets)[1], Dropout_rate=0.1)


# In[ ]:


# Builds the deep neural network
track_layers = [16 for x in range(2)]
jet_layers = [256 for x in range(3)]
b_jets_layers = [16 for x in range(2)]

track_layers = [np.shape(Tracks_input_scaled)[2]]+track_layers
print(track_layers)
print(np.shape(onehot)[1])
#Initializers the optimizer used for training the network
optimizer = tf.keras.optimizers.Nadam(LR)

#Build a DeepSets Projection Neural Network
DeepSetProjector = DeepSetsProjection(track_layers=track_layers, b_jet_layers= b_jets_layers, jet_layers=jet_layers, n_targets=np.shape(Standardized_Bhads_targets)[1], n_targets_classification= np.shape(onehot)[1], regularizer_strength= 1e-6, n_features=np.shape(b_jets_input_scaled)[1], Dropout_rate=0.01)

weights = np.array([])
for class_samples in counter.values():
    print(class_samples)
    weights = np.append(weights,len(bhads_PGID)/(4*class_samples))
print(weights)
# In[61]:


# Save the weights of the model to allow reuse in future.
path = "/home/physics/phujdj/DeepLearningParticlePhysics/CheckPointsDeepNet/ResidualDeepNetProjectorEncodedWeights&Biases.ckpt"
checkpoint_dir = os.path.dirname(path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=path,
                                                 save_weights_only=True, verbose=0, save_freq = 50*BATCHSIZE)


# In[62]:


from DeepSetNeuralNetArchitecture import LogNormal_Loss_Function
from HffragDeepSetsProjection import Multivariate_Gaussian_Negative_Likelihood_Loss_Curve
from HffragDeepSetsProjectionMultivariate import weighted_categorical_crossentropy
DeepSetProjector.compile(
 optimizer = optimizer,
 loss = {"MultivariateLoss": Multivariate_Gaussian_Negative_Likelihood_Loss_Curve, "MassOutput":weighted_categorical_crossentropy(weights)},
 metrics = [Mean_Squared_Error]   
)


#Summarises the DeepSetsProjector Set Neural Network Architecture
print(DeepSetProjector.summary())


# In[63]:


history  = DeepSetProjector.fit(
    (jets_encoded, jets_encoded), y = {"MultivariateLoss":bhads_targets_clean, "MassOutput":onehot},
    validation_split = 0.3,
    epochs = EPOCHS,
    batch_size = BATCHSIZE,
    callbacks = [early_stopping, cp_callback, reduce_learn_on_plateau],
    )


# In[ ]:


# Plot the loss and validation curves vs epoch
history_df = pd.DataFrame(history.history)
fig,ax = plt.subplots()
ax = (history_df.loc[:, ['loss', 'val_loss']]).plot()
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/ResNetDeepSetEncodedUpdatedLoss.png')
history_df.to_csv('/home/physics/phujdj/DeepLearningParticlePhysics/ResNetDeepSetEncodedUpdatedUpdated.csv')
DeepSetProjector.save_weights("/home/physics/phujdj/DeepLearningParticlePhysics/CheckPointsDeepNet/ResidualDeepSetsEncodedUpdatedWeights&Biases.ckpt")

# Output to the console the minimum epoch
print("Minimum validation loss: {}".format(history_df["val_loss"].min()))


# In[ ]:



#Evaluate the entire performance of the model
loss = DeepSetProjector.evaluate((tracks_encoded, jets_encoded),(bhads_targets_clean, onehot),verbose = 2)
print("The Loaded DeepNet has loss: ", loss)


# In[ ]:

tracks_encoded_unmasked = tracks_encoded
jets_encoded_unmasked = jets_encoded


# In[ ]:


#Evaluate the entire performance of the model
loss = DeepSetProjector.evaluate((tracks_encoded_unmasked, jets_encoded_unmasked),(bhads_targets_clean, onehot),verbose = 2)
print("The Loaded DeepNet has loss: ", loss)


# In[ ]:


predictions_mass = DeepSetProjector.predict((tracks_encoded_unmasked, jets_encoded_unmasked))[1]


# In[200]:

Predictions = np.stack([DeepSetProjector((tracks_encoded_unmasked, jets_encoded_unmasked))[0] for sample in range(100)])
Predictions = Predictions.mean(axis = 0)


# In[ ]:

Predictions_Mass = predictions_mass



# In[ ]:


# In[202]:

Predictions_X = Predictions[:,:np.shape(Standardized_Bhads_targets)[1]]
predictions_x_uncertainty = Predictions[:,np.shape(Standardized_Bhads_targets)[1]:2*np.shape(Standardized_Bhads_targets)[1]]
Predictions_Uncertainty = predictions_x_uncertainty
print(np.shape(Predictions_Uncertainty))

# In[ ]:


# In[ ]:
predictions_X = Predictions[:,:np.shape(Standardized_Bhads_targets)[1]]
predictions_x_uncertainty = Predictions[:,np.shape(Standardized_Bhads_targets)[1]:2*np.shape(Standardized_Bhads_targets)[1]]
Predictions_Uncertainty = predictions_x_uncertainty
print(np.shape(Predictions_Uncertainty))


# In[204]:


Error_px_unscaled = (bhads_targets_clean[:,0] - predictions_X[:,0])*1e6
Pull_bhads_px_unscaled = Error_px_unscaled/np.std(bhads_targets_clean[:,0]*1e6)
Pull_px = Error_px_unscaled/(Predictions_Uncertainty[:,0]*1e6)


# In[ ]:


Error_py_unscaled = (bhads_targets_clean[:,1] - predictions_X[:,1])*1e6
Pull_bhads_py_unscaled = Error_py_unscaled/np.std(bhads_targets_clean[:,1]*1e6)
Pull_py = Error_py_unscaled/(Predictions_Uncertainty[:,1]*1e6)


# In[ ]:


Error_pz_unscaled = (bhads_targets_clean[:,2] - predictions_X[:,2])*1e6
Pull_bhads_pz_unscaled = Error_pz_unscaled/np.std(bhads_targets_clean[:,2]*1e6)
Pull_pz = Error_pz_unscaled/(Predictions_Uncertainty[:,2]*1e6)


# In[69]:


Error_pt_unscaled = (bhads_targets_clean[:,3] - predictions_X[:,3])*1e6
Pull_bhads_pt_unscaled = Error_pt_unscaled/np.std(bhads_targets_clean[:,3]*1e6)
Pull_pt = Error_pt_unscaled/(Predictions_Uncertainty[:,3]*1e6)


# In[70]:


Error_eta_unscaled = bhads_targets_clean[:,4] - predictions_X[:,4]
Pull_bhads_eta_unscaled = Error_eta_unscaled/np.std(bhads_targets_clean[:,4])
Pull_eta = Error_eta_unscaled/Predictions_Uncertainty[:,4]


# In[71]:


Error_phi_unscaled = bhads_targets_clean[:,5] - predictions_X[:,5]
Pull_bhads_pt_unscaled = Error_pt_unscaled/np.std(bhads_targets_clean[:,5])
Pull_phi = Error_phi_unscaled/Predictions_Uncertainty[:,5]


# In[72]:


Error_pxfraction_unscaled = bhads_targets_clean[:,6] - predictions_X[:,6]
Pull_pxfraction_unscaled = Error_pxfraction_unscaled/np.std(bhads_targets_clean[:,6])
Pull_pxfraction = Error_pxfraction_unscaled/Predictions_Uncertainty[:,6]


# In[73]:


Error_pyfraction_unscaled = bhads_targets_clean[:,7] - predictions_X[:,7]
Pull_pyfraction_unscaled = Error_pyfraction_unscaled/np.std(bhads_targets_clean[:,7])
Pull_pyfraction = Error_pyfraction_unscaled/Predictions_Uncertainty[:,7]


# In[74]:


Error_pzfraction_unscaled = bhads_targets_clean[:,8] - predictions_X[:,8]
Pull_pzfraction_unscaled = Error_pzfraction_unscaled/np.std(bhads_targets_clean[:,8])
Pull_pzfraction = Error_pzfraction_unscaled/Predictions_Uncertainty[:,8]


# In[77]:


Error_energy_unscaled = (bhads_targets_clean[:,9] - predictions_X[:,9])*1e6
Pull_energy_unscaled = Error_energy_unscaled/np.std(bhads_targets_clean[:,9]*1e6)
Pull_energy = Error_energy_unscaled/(Predictions_Uncertainty[:,9]*1e6)


# In[76]:


Error_projection_unscaled = bhads_targets_clean[:,12] - predictions_X[:,12]
Pull_bhads_projection_unscaled = Error_projection_unscaled/np.std(bhads_targets_clean[:,12])
Pull_projection = Error_projection_unscaled/Predictions_Uncertainty[:,12]


# In[78]:


Error_mass_unscaled = (bhads_targets_clean[:,13] - predictions_X[:, 13])*1e3
Pull_mass_unscaled = Error_energy_unscaled/np.std(bhads_targets_clean[:,13]*1e3)
Pull_mass = Error_energy_unscaled/(Predictions_Uncertainty[:,13]*1e3)

Error_Secondary_Displacement_unscaled = (bhads_targets_clean[:,14] - predictions_X[:, 14])*1e4
Pull_Secondary_Displacement_unscaled = Error_Secondary_Displacement_unscaled/np.std(bhads_targets_clean[:,14]*1e4)
Pull_Secondary_Displacement = Error_Secondary_Displacement_unscaled/(Predictions_Uncertainty[:,14]*1e4)


# In[ ]:


print(np.mean(Error_px_unscaled))
print(np.std(Error_px_unscaled))
fig = binneddensity(Error_px_unscaled, fixedbinning(-20000,20000,100),xlabel = "Error_px_unscaled")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/Error_PxResNetDeepSetsEncodedUpdated.png')
plt.close()

# In[208]:


print(np.mean(Pull_px))
print(np.std(Pull_px))
fig = binneddensity(Pull_px, fixedbinning(-2,2,100),xlabel = "Pull_Px")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/Pull_Px_ResNetDeepSetsEncodedUpdated.png')
plt.close()

# In[209]:


print(np.mean(Error_py_unscaled))
print(np.std(Error_py_unscaled))
fig = binneddensity(Error_py_unscaled, fixedbinning(-20000,20000,100),xlabel = "Error_py_unscaled")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/Error_Py_ResNetDeepSetsEncodedUpdated.png')
plt.close()

# In[210]:


print(np.mean(Pull_py))
print(np.std(Pull_py))
fig = binneddensity(Pull_py, fixedbinning(-2,2,100),xlabel = "Pull_Py")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/Pull_Py_ResNetDeepSetsEncodedUpdated.png')
plt.close()

# In[211]:


print(np.mean(Error_pz_unscaled))
print(np.std(Error_pz_unscaled))
fig = binneddensity(Error_pz_unscaled, fixedbinning(-40000,40000,100),xlabel = "Error_pz_unscaled")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/Error_Pz_ResNetDeepSetsEncodedUpdated.png')
plt.close()

# In[212]:


print(np.mean(Pull_pz))
print(np.std(Pull_pz))
fig = binneddensity(Pull_pz, fixedbinning(-2,2,100),xlabel = "Pull_Pz")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/Pull_Pz_ResNetDeepSetsEncodedUpdated.png')
plt.close()

# In[213]:


print(np.mean(Error_projection_unscaled))
print(np.std(Error_projection_unscaled))
fig = binneddensity(Error_projection_unscaled, fixedbinning(-0.01,0.01,100),xlabel = "Error_projection")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/Error_Projection_ResNetDeepSetsEncodedUpdated.png')
plt.close()

# In[216]:


print(np.mean(Pull_projection))
print(np.std(Pull_projection))
fig = binneddensity(Pull_projection, fixedbinning(-2,2,100),xlabel = "Pull_projection")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/Pull_Projection_ResNetDeepSetsEncodedUpdated.png')
plt.close()

# In[217]:


print(np.mean(Error_pxfraction_unscaled))
print(np.std(Error_pxfraction_unscaled))
fig = binneddensity(Error_pxfraction_unscaled, fixedbinning(-1,1,100),xlabel = "Error_px_fraction")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/Error_px_fraction_ResNetDeepSetsEncodedUpdated.png')
plt.close()

# In[218]:


print(np.mean(Pull_pxfraction_unscaled))
print(np.std(Pull_pxfraction_unscaled))
fig = binneddensity(Pull_pxfraction_unscaled, fixedbinning(-2,2,100),xlabel = "Pull_px_fraction")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/Pull_px_fraction_ResNetDeepSetsEncodedUpdated.png')
plt.close()

# In[219]:


print(np.mean(Error_pyfraction_unscaled))
print(np.std(Error_pyfraction_unscaled))
fig = binneddensity(Error_pyfraction_unscaled, fixedbinning(-1,1,100),xlabel = "Error_py_fraction")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/Error_py_fraction_ResNetDeepSetsEncodedUpdated.png')
plt.close()

# In[220]:


print(np.mean(Pull_pyfraction_unscaled))
print(np.std(Pull_pyfraction_unscaled))
fig = binneddensity(Pull_pyfraction_unscaled, fixedbinning(-1,1,100),xlabel = "Pull_py_fraction")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/Pull_py_fraction_ResNetDeepSetsEncodedUpdated.png')
plt.close()

# In[221]:


print(np.mean(Error_pzfraction_unscaled))
print(np.std(Error_pzfraction_unscaled))
fig = binneddensity(Error_pzfraction_unscaled, fixedbinning(-1,1,100),xlabel = "Error_pz_fraction")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/Error_pz_fraction_ResNetDeepSetsEncodedUpdated.png')
plt.close()

# In[96]:


print(np.mean(Pull_pzfraction_unscaled))
print(np.std(Pull_pzfraction_unscaled))
fig = binneddensity(Pull_pzfraction_unscaled, fixedbinning(-2,2,100),xlabel = "Pull_pz_fraction")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/Pull_pz_fraction_ResNetDeepSetsEncodedUpdated.png')
plt.close()

# In[222]:


print(np.mean(Error_energy_unscaled))
print(np.std(Error_energy_unscaled))
fig = binneddensity(Error_energy_unscaled, fixedbinning(-40000,40000,100),xlabel = "Error_energy")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/Error_energy_ResNetDeepSetsEncodedUpdated.png')
plt.close()

# In[223]:


print(np.mean(Pull_energy_unscaled))
print(np.std(Pull_energy_unscaled))
fig = binneddensity(Pull_energy_unscaled, fixedbinning(-1,1,100),xlabel = "Pull_energy")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/Pull_energy_ResNetDeepSetsEncodedUpdated.png')
plt.close()

# In[224]:


print(np.mean(Error_mass_unscaled))
print(np.std(Error_mass_unscaled))
fig = binneddensity(Error_mass_unscaled, fixedbinning(-250,250,100),xlabel = "Mass Error")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/Mass_Error_ResNetDeepSetsEncodedUpdated.png')
plt.close()

# In[225]:


print(np.mean(Error_pt_unscaled))
print(np.std(Error_pt_unscaled))
fig = binneddensity(Error_pt_unscaled, fixedbinning(-40000,40000,100),xlabel = "Error_pt")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/Error_pt_ResNetDeepSetsEncodedUpdated.png')
plt.close()


# In[227]:


print(np.mean(Pull_pt))
print(np.std(Pull_pt))
fig = binneddensity(Pull_pt, fixedbinning(-2,2,100),xlabel = "Pull_pt")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/Pull_pt_ResNetDeepSetsEncodedUpdated.png')
plt.close()


# In[228]:


print(np.mean(Error_eta_unscaled))
print(np.std(Error_eta_unscaled))
fig = binneddensity(Error_eta_unscaled, fixedbinning(-0.1,0.1,100),xlabel = "Error_eta")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/Error_eta_ResNetDeepSetsEncodedUpdated.png')
plt.close()

# In[229]:


print(np.mean(Pull_eta))
print(np.std(Pull_eta))
fig = binneddensity(Pull_eta, fixedbinning(-3,3,100),xlabel = "Pull_eta")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/Pull_eta_ResNetDeepSetsEncodedUpdated.png')
plt.close()

# In[106]:


print(np.mean(Error_phi_unscaled))
print(np.std(Error_phi_unscaled))
fig = binneddensity(Error_phi_unscaled, fixedbinning(-0.1,0.1,100),xlabel = "Error_phi")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/Error_phi_ResNetDeepSetsEncodedUpdated.png')
plt.close()

# In[107]:


print(np.mean(Pull_phi))
print(np.std(Pull_phi))
fig = binneddensity(Pull_phi, fixedbinning(-3,3,100),xlabel = "Pull_phi")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/Pull_phi_ResNetDeepSetseEnodedUpdated.png')
plt.close()

# In[108]:


print(np.median(Error_Secondary_Displacement_unscaled))
print(np.std(Error_Secondary_Displacement_unscaled))
fig = binneddensity(Error_Secondary_Displacement_unscaled, fixedbinning(-10,10,100),xlabel = "Error_SecondaryVertex")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/Error_SecondaryVertex_ResNetDeepSetsEncodedUpdated.png')
plt.close()

# In[109]:


print(np.median(Pull_Secondary_Displacement))
print(np.std(Pull_Secondary_Displacement))
fig = binneddensity(Pull_Secondary_Displacement, fixedbinning(-1,1,100),xlabel = "Pull_SecondaryVertex")
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/Pull_SecondaryVertex_ResNetDeepSetsEncodedUpdated.png')
plt.close()

# In[ ]:


import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
fig, ax = plt.subplots(figsize = (12,12))
x = bhads_targets_clean[:,0]*1e6
y = predictions_X[:,0]*1e6
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
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/PxScatter_ResNetDeepSetsEncodedUpdated.png')
plt.close()

# In[ ]:


[bhads[:,0],bhads[:,1], bhads[:,2], bhads_pt, bhads_eta, bhads_phi, bhads_fractions_px, bhads_fractions_py, bhads_fractions_pz, bhads_energy, bhads_transverse_mass, bhads_energy_mass_ratio, bhads_projection, bhads_m, Secondary_Displacement]


# In[ ]:


import matplotlib as mpl
from scipy.stats import gaussian_kde
fig, ax = plt.subplots(figsize = (12,12))
x = bhads_targets_clean[:,1]*1e6
y = predictions_X[:,1]*1e6
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
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/PyScatter_ResNetDeepSetsEncodedUpdated.png')
plt.close()

# In[ ]:


import matplotlib as mpl
from scipy.stats import gaussian_kde
fig, ax = plt.subplots(figsize = (12,12))
x = bhads_targets_clean[:,2]*1e6
y = predictions_X[:,2]*1e6
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
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/PzScatter_ResNetDeepSetsEncodedUpdated.png')
plt.close()

# In[ ]:


import matplotlib as mpl
from scipy.stats import gaussian_kde
fig, ax = plt.subplots(figsize = (12,12))
x = bhads_targets_clean[:,3]*1e6
y = predictions_X[:,3]*1e6
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
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/PtScatter_ResNetDeepSetsEncodedUpdated.png')
plt.close()

# In[ ]:


import matplotlib as mpl
from scipy.stats import gaussian_kde
fig, ax = plt.subplots(figsize = (12,12))
x = bhads_targets_clean[:,4]
y = predictions_X[:,4]
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
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/PsuedorapidityScatter_ResNetDeepSetsEncodedUpdated.png')
plt.close()

# In[ ]:


import matplotlib as mpl
from scipy.stats import gaussian_kde
fig, ax = plt.subplots(figsize = (12,12))
x = bhads_targets_clean[:,5]
y = predictions_X[:,5]
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
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/PhiScatter_ResNetDeepSetsEncodedUpdated.png')
plt.close()

# In[ ]:


import matplotlib as mpl
from scipy.stats import gaussian_kde
fig, ax = plt.subplots(figsize = (12,12))
x = bhads_targets_clean[:,6]
y = predictions_X[:,6]
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
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/PxMomentaFraction_Scatter_ResNetDeepSetsEncodedUpdated.png')
plt.close()

# In[ ]:


import matplotlib as mpl
from scipy.stats import gaussian_kde
fig, ax = plt.subplots(figsize = (12,12))
x = bhads_targets_clean[:,7]
y = predictions_X[:,7]
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
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/PyMomentaFraction_Scatter_ResNetDeepSetsEncodedUpdated.png')
plt.close()


# In[ ]:


import matplotlib as mpl
from scipy.stats import gaussian_kde
fig, ax = plt.subplots(figsize = (12,12))
x = bhads_targets_clean[:,8]
y = predictions_X[:,8]
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
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/PzMomentaFraction_Scatter_ResNetDeepSetsEncodedUpdated.png')
plt.close()


# In[ ]:


import matplotlib as mpl
from scipy.stats import gaussian_kde
fig, ax = plt.subplots(figsize = (12,12))
x = bhads_targets_clean[:,9]*1e6
y = predictions_X[:,9]*1e6
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
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/BhadronEnergyMomentaFraction_Scatter_ResNetDeepSetsEncodedUpdated.png')
plt.close()


# In[ ]:


import matplotlib as mpl
from scipy.stats import gaussian_kde
fig, ax = plt.subplots(figsize = (12,12))
x = bhads_targets_clean[:,12]
y = predictions_X[:,12]
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
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/ProjectionFraction_Scatter_ResNetDeepSetsEncodedUpdated.png')
plt.close()


# In[ ]:


import matplotlib as mpl
from scipy.stats import gaussian_kde
fig, ax = plt.subplots(figsize = (12,12))
x = bhads_targets_clean[:,13]*1e3
y = predictions_X[:,13]*1e3
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
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/Mass_Scatter_ResNetDeepSetsEncodedUpdated.png')
plt.close()

# In[ ]:


import matplotlib as mpl
from scipy.stats import gaussian_kde
fig, ax = plt.subplots(figsize = (12,12))
x = bhads_targets_clean[:,14]*1e4
y = predictions_X[:,14]*1e4
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
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/SV_Scatter_ResNetDeepSetsEncodedUpdated.png')
plt.close()

# In[110]:


from sklearn.metrics import classification_report
print(classification_report(np.argmax(onehot, axis = 1), np.argmax(Predictions_Mass,axis = 1)))


# In[111]:


from sklearn.metrics import confusion_matrix
from sklearn import metrics
confusion_mat = confusion_matrix(np.argmax(onehot, axis = 1), np.argmax(Predictions_Mass,axis = 1))
confusion_visualized = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_mat)
fig =confusion_visualized.plot()
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/ConfusionMatrixResNetDeepSetsEncodedUpdated.png')
plt.close()

# In[ ]:


print(predictions_X[:10,13])
print(bhads_targets_clean[:10,13])


import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import RocCurveDisplay
target_names = ["521/-521", "511/-511", "531/-531", "Other"]
fig, ax = plt.subplots(figsize = (6,6))
colors = cycle(["aqua", "darkorange", "cornflowerblue"])
for class_id, colors in zip(range(4), colors):
    RocCurveDisplay.from_predictions(
    onehot[:,class_id],
    Predictions_Mass[:, class_id],
    name = f"ROC curve for {target_names[class_id]}",
    color = colors,
    ax = ax,)
plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("One-vs-Rest ROC curves:\n Bhadron Masses vs the Rest:")
plt.legend()
fig.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/Results/ROCCurveResNetDeepSetsEncodedUpdated.png')
plt.close()

# In[ ]:



