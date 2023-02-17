
"""
*Filename: hffragTrainer
*Description: This is python implementation of the jupyter notebooks 
*hffragTrainer that takes the hffragDeepSets architecture and trains it.
Date: 16/02/2023
Author: Kaylen Smith Darnbrook
"""
#Starts by importing the relevant python modules
import uproot
import os
import awkward as ak
import sklearn as sk
import seaborn as sns
import pandas as pd
import numpy as np
import tensorflow as tf
from keras import callbacks
import keras
from hffrag import fixedbinning
from hffrag import binneddensity
import DeepSetNeuralNetArchitecture as DSNNA
import hffragJetNetwork as hffragJN
from sklearn.feature_selection import mutual_info_regression
from numpy.lib.recfunctions import structured_to_unstructured
from sklearn.model_selection import train_test_split
from hffragJetNetwork import LogNormal_Loss_Function
from timeit import default_timer as timer
import matplotlib.pyplot as plt

#Format the style and format of the matplotlib plots
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'red'
plt.rc('text',usetex = False)
plt.rc('font',family = 'Times New Roman')

# The data is being stored in a tree datastructure.
# We access the charm root using this command
tree = uproot.open("hffrag.root:CharmAnalysis")

# Initial parameters
MASKVAL = -999 # This value is introduced to ensure arrays are regular (Of the same size). They will be masked later by the network
MAXTRACKS = 32 # This value is the maximum number of tracks allowed per event
BATCHSIZE = 128 # This is the batch size of the mini batches used during training
EPOCHS = 2000 # This is the default number of epochs for which the neural network will train providing that early stopping does not occur
MAXEVENTS = 1e20 #This is the maximum number of events that will the program will accept
LR = 1e-5 #This is the default learning rate

def make_mi_scores(X,y):
    mi_scores = mutual_info_regression(X,y)
    mi_scores = pd.Series(mi_scores,name = "MI Scores")
    mi_scores = mi_scores.sort_values(ascending = False)
    return mi_scores
def plot_mi_scores(scores):
    scores = scores.sort_values(ascending = True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width,ticks)
    plt.title("Mutual Informarion Scores")

# Select the features we wish to study
track_features = ["AnalysisTracks_pt", "AnalysisTracks_eta", "AnalysisTracks_phi", "AnalysisTracks_z0sinTheta",
                  "AnalysisTracks_d0sig", "AnalysisTracks_d0", "AnalysisTracks_d0sigPV", "AnalysisTracks_d0PV"]
jet_features = ["AnalysisAntiKt4TruthJets_pt", "AnalysisAntiKt4TruthJets_eta", "AnalysisAntiKt4TruthJets_phi",
                "AnalysisAntiKt4TruthJets_ghostB_pt", "AnalysisAntiKt4TruthJets_ghostB_eta","AnalysisAntiKt4TruthJets_ghostB_phi"]

# Read in the data from the root file
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

bhads = np.stack([bhads_pt,bhads_eta,bhads_phi],axis = -1) #Combine the momentum, eta and phi for each jet into one array

print("There are {} outputs".format(np.shape(bhads)[1])) # Display the number of target features the neural network will predict
matchedtracks = matchedtracks[bjets]
print("There are {} inputs".format(np.shape(matchedtracks)[1])) # Display the number of target features the neural network will use in it's predictions

jets_pt = jets["AnalysisAntiKt4TruthJets_pt"].to_numpy()
jets_eta = jets["AnalysisAntiKt4TruthJets_eta"].to_numpy()
jets_phi = jets["AnalysisAntiKt4TruthJets_phi"].to_numpy()
b_jets = np.stack([jets_pt,jets_eta,jets_phi], axis = -1)

# Transform the jet and tracks to unstructed data.
jets = structured_to_unstructured(jets[jet_features[:-3]])
matchedtracks = structured_to_unstructured(matchedtracks)

# Convert the coordinates of the b jets and tracks to cartesian coordinates
tracks_p = DSNNA.pt_eta_phi_2_px_py_pz_tracks(matchedtracks.to_numpy())
bhads = DSNNA.pt_eta_phi_2_px_py_pz_jets(bhads)
b_jets = DSNNA.pt_eta_phi_2_px_py_pz_jets(b_jets)

#Combine the momenta of the tracks with the rest of the track features to form the track dataset
tracks = np.concatenate([tracks_p,matchedtracks[:,:,3:].to_numpy()],axis = 2)

sum_of_tracks_px = np.sum(tracks[:,:,0],axis = -1)
sum_of_tracks_py = np.sum(tracks[:,:,1],axis = -1)
sum_of_tracks_pz = np.sum(tracks[:,:,2],axis = -1)

sum_px = sum_of_tracks_px.reshape(sum_of_tracks_pz.shape[0],1)
sum_py = sum_of_tracks_py.reshape(sum_of_tracks_py.shape[0],1)
sum_pz = sum_of_tracks_pz.reshape(sum_of_tracks_px.shape[0],1)

print(sum_py.shape)
print(sum_px.shape)
print(sum_pz.shape)

b_jets_new = np.concatenate([b_jets,sum_px,sum_py,sum_pz],axis = -1)
print(b_jets_new.shape)

bhads_scaled = bhads/b_jets
print(bhads_scaled[0])

# Split the data into training and validation sets.
X_train, X_valid, y_train, y_valid = train_test_split(
    tracks, bhads_scaled, train_size=0.8, random_state=42)

jet_embedder_layers = [200,200,200,200,200]
n_targets = 3
n_features = 6

sample_bhadNet = hffragJN.bhadsNet(n_features, n_targets, NumDropout=0,jet_layers=jet_embedder_layers,Dropout_rate=0.001)

print(sample_bhadNet.summary())

sample_bhadNet.compile(
    optimizer = tf.keras.optimizers.Nadam(0.00001,beta_1=0.9, beta_2 = 0.999, epsilon= 1e-8),
    loss = {"Target_Values": tf.keras.losses.MeanSquaredError(), "Jet_Out":LogNormal_Loss_Function}
)

class TimingCallback(keras.callbacks.Callback):
    def __init__(self, logs = {}):
        self.logs = []
    def on_epoch_begin(self, epoch, logs ={}):
        self.starttime = timer()
    def on_epoch_end(self, epoch, logs = {}):
        self.logs.append(timer() - self.starttime)

# Introduce early_stopping to prevent overfitting
early_stopping = callbacks.EarlyStopping(
    min_delta=0.00001,  # The minimum amount of change to count as an improvement
    patience=30,  # The number of epochs to wait before stopping
    restore_best_weights=True,  # Keep the best weights
)

# Prevent spikes in the validation and training loss due to the gradient descent kicking the network out of a local minima
reduce_learn_on_plateau = callbacks.ReduceLROnPlateau(
    monitor='loss', factor=0.80, patience=5, min_lr=1e-8)

# Save the weights of the model to allow reuse in future.
path = "/home/physics/phujdj/DeepLearningParticlePhysics/CheckPointsDeepNet/DeepNetWeights&Biases.ckpt"
checkpoint_dir = os.path.dirname(path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=path,
                                                 save_weights_only=True, verbose=0, save_freq = 100*BATCHSIZE)
#Timer
cb = TimingCallback()


history = sample_bhadNet.fit(
    b_jets_new,[bhads,bhads],
    validation_data = (b_jets_new,[bhads,bhads]),
    callbacks = [early_stopping,reduce_learn_on_plateau,hffragJN.PredictOnEpoch(sample_bhadNet,b_jets_new),cb,cp_callback],
    batch_size=BATCHSIZE,
    epochs = EPOCHS,
)

#Evaluate the entire performance of the model
loss = sample_bhadNet.evaluate(X_train,y_train,verbose = 2)
print("The Loaded sample_bhadNet has loss: ", loss)

# Plot the loss and validation curves vs epoch
history_df = pd.DataFrame(history.history)
LogLossPlot = np.log(history_df.loc[:, ["loss","val_loss"]]).plot()
history_df.to_csv('/home/physics/phujdj/DeepLearningParticlePhysics/history.csv')
LogLossPlot.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/LogLossPlot.png')

print(sum(cb.logs))

# Output to the console the minimum epoch
print("Minimum validation loss: {}".format(history_df["loss"].min()))

#Obtain predictions and plot the associated resolutions, pulls and pull bhads.
PredictionsMeans, PredictionsCovs = sample_bhadNet.predict(X_train)
print(PredictionsMeans.shape)
print(PredictionsCovs.shape)

mi_scores_px_momentum =  make_mi_scores(PredictionsCovs[:,0], bhads[:,0])
print(mi_scores_px_momentum)

mi_scores_py_momentum =  make_mi_scores(PredictionsCovs[:,1], bhads[:,0])
print(mi_scores_py_momentum)

mi_scores_pz_momentum =  make_mi_scores(PredictionsCovs[:,2], bhads[:,0])
print(mi_scores_pz_momentum)

Error_px = y_train[:,0] - PredictionsCovs[:,0]
Pull_px = Error_px/PredictionsCovs[:,3]

Error_py = y_train[:,1] - PredictionsCovs[:,1]
Pull_py = Error_py/PredictionsCovs[:,4]

Error_pz = y_train[:,2] - PredictionsCovs[:,2]
Pull_pz = Error_pz/PredictionsCovs[:,5]

print(np.mean(Error_px))
print(np.std(Error_px))
Errorplot = binneddensity(Error_px, fixedbinning(-400000,400000,100),xlabel = "Error_px")
Errorplot.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/ErrorPlot_px.png')

print(np.mean(Error_py))
print(np.std(Error_py))
Errorplot = binneddensity(Error_py, fixedbinning(-400000,400000,100),xlabel = "Error_py")
Errorplot.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/ErrorPlot_py.png')

print(np.mean(Error_pz))
print(np.std(Error_pz))
Errorplot = binneddensity(Error_pz, fixedbinning(-400000,400000,100),xlabel = "Error_pz")
Errorplot.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/ErrorPlot_pz.png')

print(np.mean(Pull_px))
print(np.std(Pull_px))
Pullplot = binneddensity(Pull_px, fixedbinning(-3,3,100),xlabel = "Pull_px")
Pullplot.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/PullPlot_px.png')

print(np.mean(Pull_py))
print(np.std(Pull_py))
Pullplot = binneddensity(Pull_px, fixedbinning(-3,3,100),xlabel = "Pull_py")
Pullplot.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/PullPlot_py.png')

print(np.mean(Pull_pz))
print(np.std(Pull_pz))
Pullplot = binneddensity(Pull_px, fixedbinning(-3,3,100),xlabel = "Pull_pz")
Pullplot.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/PullPlot_pz.png')

fig,ax = plt.subplots(figsize = (12,12))
sns.scatterplot(
    y = PredictionsCovs[:,0],
    x = y_train[:,0],
    color = "blue"
)
ax.set_title("Scatterplot of the true vs pred X momenta")
ax.set_xlabel("The true X momenta of the tracks from each event")
ax.set_ylabel("The predicted X momenta of b hadron jets")

fig,ax = plt.subplots(figsize = (12,12))
sns.scatterplot(
    y = PredictionsCovs[:,1],
    x = y_train[:,1],
    color = "red"
)
ax.set_title("Scatterplot of the true vs pred Y momenta")
ax.set_xlabel("The true Y momenta of the tracks from each event")
ax.set_ylabel("The predicted Y momenta of b hadron jets")

fig,ax = plt.subplots(figsize = (12,12))
sns.scatterplot(
    y = PredictionsCovs[:,2],
    x = y_train[:,2],
    color = "green"
)
ax.set_title("Scatterplot of the true vs pred Z momenta")
ax.set_xlabel("The true Z momenta of the tracks from each event")
ax.set_ylabel("The predicted Z momenta of b hadron jets")

from DeepSetNeuralNetArchitecture import Predicted_Bhad_px
from DeepSetNeuralNetArchitecture import Predicted_Bhad_px_uncertainties
print(Predicted_Bhad_px.shape)
print(Predicted_Bhad_px_uncertainties.shape)
Predicted_Bhad_px = Predicted_Bhad_px.reshape(-1,y_train.shape[0])
Predicted_Bhad_px_uncertainties = Predicted_Bhad_px_uncertainties.reshape(-1,y_train.shape[0])
print(Predicted_Bhad_px.shape)
print(Predicted_Bhad_px_uncertainties.shape)

Mean_Predictions = np.mean(Predicted_Bhad_px,axis = 1)
Std_Predictions = np.std(Predicted_Bhad_px, axis = 1)

Mean_Predictions_Uncertainities = np.mean(Predicted_Bhad_px_uncertainties, axis = 1)
Std_Predictions_Uncertaintiies = np.std(Predicted_Bhad_px_uncertainties, axis = 1)

Scaled_Mean_Squared_Error = (Mean_Predictions - np.mean(y_train[:,0], axis = 0))/(Std_Predictions_Uncertaintiies)

True_X_Momentum = np.full(np.shape(Mean_Predictions),fill_value = np.mean(y_train[:,0]))
Std_X_Momentum =np.full(np.shape(Mean_Predictions),fill_value = np.std(y_train[:,0])) 
print(True_X_Momentum.shape)

Figure, axes = plt.subplots()
axes.set_xlabel("Epoch Number")
axes.set_ylabel("Px Momentum")
Pred = axes.plot(Mean_Predictions)
StdPred = axes.plot(Std_Predictions)
SMSE = axes.plot(Scaled_Mean_Squared_Error)
TXM = axes.plot(True_X_Momentum)
STXM = axes.plot(Std_X_Momentum)
#axes.legend([Pred,StdPred,SMSE,TXM,STXM], ["The Mean Predicted X Momentum", "The Std Predicted X Momentum", "The ScaledMeanSquareError","TrueMeanXMometum" "TrueStdXMomentum"])
Figure.savefig('/home/physics/phujdj/DeepLearningParticlePhysics/PredictionsOverEpoch.png')


