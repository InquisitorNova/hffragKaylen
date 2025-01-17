#!/usr/bin/env python3
# coding: utf-8

# In[100]:


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
import matplotlib as mpl

#Update the maximum chunksize
mpl.rcParams['agg.path.chunksize'] = 10000


# In[103]:


# Initial parameters
MASKVAL = -999 # This value is introduced to ensure arrays are regular (Of the same size). They will be masked later by the network
MAXTRACKS = 32 # This value is the maximum number of tracks allowed per event
BATCHSIZE = 64 # This is the batch size of the mini batches used during training
EPOCHS = 200 # This is the default number of epochs for which the neural network will train providing that early stopping does not occur
MAXEVENTS = 1e20 #This is the maximum number of events that will the program will accept
LR = 1e-4 #This is the default learning rate

#Create global variables to store the predictions per epoch
Predicted_Bhad_px = np.array([])
Predicted_Bhad_px_uncertainties = np.array([])

# In[104]:


# Find the associated tracks for each jet
def Match_Tracks(jets, tracks):
    """Used to determine if a set of tracks belong to a particular set of jets"""

    jet_eta = jets["AnalysisAntiKt4TruthJets_eta"]
    jet_phi = jets["AnalysisAntiKt4TruthJets_phi"] 

    tracks_eta = tracks["AnalysisTracks_eta"]
    tracks_phi = tracks["AnalysisTracks_phi"]

    delta_etas = jet_eta - tracks_eta
    delta_phis = np.abs(jet_phi - tracks_phi)

    # Map the phis from a cyclical period onto a linear relation
    ak.where(delta_phis > np.pi, delta_phis - np.pi, delta_phis)

    # Returns a list of true and false, determining which tracks belong to those jets.
    return np.sqrt(delta_phis**2 + delta_etas**2) < 0.4

#Finds the associated electrons for each jet
def Match_Electrons(jets, electrons):
    "Used to determine if a set of electrons belong to a particlular set of jets"

    jet_eta = jets["AnalysisAntiKt4TruthJets_eta"]
    jet_phi = jets["AnalysisAntiKt4TruthJets_phi"]

    electron_eta =  electrons["AnalysisElectrons_eta"]
    electron_phi = electrons["AnalysisElectrons_phi"]
    
    delta_etas = jet_eta - electron_eta
    delta_phis = np.abs(jet_phi - electron_phi)


    # Map the phis from a cyclical period onto a linear relation
    ak.where(delta_phis > np.pi, delta_phis - np.pi, delta_phis)

    # Returns a list of true and false, determining which tracks belong to those jets.
    return np.sqrt(delta_phis**2 + delta_etas**2) < 0.4

#Finds the associated muons for each jet
def Match_Muons(jets, muons):
    "Used to determine if a set of muons belong to a particular set of muons"

    jet_eta = jets["AnalysisAntiKt4TruthJets_eta"]
    jet_phi = jets["AnalysisAntiKt4TruthJets_phi"]

    muon_eta = muons["AnalysisMuons_eta"]
    muon_phi = muons["AnalysisMuons_phi"]

    delta_etas = jet_eta - muon_eta
    delta_phis = np.abs(jet_phi - muon_phi)

    # Map the phis from a cyclical period onto a linear relation
    ak.where(delta_phis > np.pi, delta_phis - np.pi, delta_phis)

    # Returns a list of true and false, determining which tracks belong to those jets.
    return np.sqrt(delta_phis**2 + delta_etas**2) < 0.4

# In[105]:


# Convert from cylindrical to cartesian coordinates
def pt_eta_phi_2_px_py_pz_jets(pt_eta_phi):
    """Converts the cylindrical polar coordinates to cartesian coordinates for jets"""

    # Seperate the pts, etas and phis
    pts = pt_eta_phi[:, 0:1]
    etas = pt_eta_phi[:, 1:2]
    phis = pt_eta_phi[:, 2:3]

    # Convert from polar to cartesian
    pxs = pts * np.cos(phis)
    pys = pts * np.sin(phis)
    pzs = pts * np.sinh(etas)

    # Check to see if there are any infinities
    isinf = np.isinf(pzs)

    if np.any(isinf):
        print("Infinities in eta detected!")
        print(etas[isinf])
        raise ValueError("Infinity from sinh(eta) has been detected")

    # Returns the momentum vector
    return np.concatenate([pxs, pys, pzs], axis=1)


# In[106]:


def pt_eta_phi_2_px_py_pz_tracks(pt_eta_phi, MASKVAL=-999):
    """Converts the cylindrical polar coordinates to cartesian coordinates for jets"""

    # Seperate the pts, etas and phis
    pts = pt_eta_phi[:, :, 0:1]
    etas = pt_eta_phi[:, :, 1:2]
    phis = pt_eta_phi[:, :, 2:3]

    # Convert from polar to cartesian
    # Transforms only the non masked values from cylindrical to cartesian coordinates. Mask values are left unchanged.
    mask1 = pts == MASKVAL 
    mask2 = phis == MASKVAL
    mask3 = etas == MASKVAL
    pxs = np.where(mask1 | mask2, pts, pts * np.cos(phis)) 
    pys = np.where(mask1 | mask2, pts, pts * np.sin(phis))
    pzs = np.where(mask1 | mask3, pts, pts * np.sinh(etas))

    # Check to see if there are any infinities
    isinf = np.isinf(pzs)

    if np.any(isinf):
        print("Infinities in eta detected!")
        print(etas[isinf])
        raise ValueError("Infinity from sinh(eta) has been detected")

    # Returns the momentum vector in cartesian coordinates
    return np.concatenate([pxs, pys, pzs], axis=2)


# In[107]:


def pt_eta_phi2_px_py_pz_predicted_tracks(predictions):
    #Obtain the pts,etas and phis
    pts = predictions[:,0:1]
    etas = predictions[:,1:2]
    phis = predictions[:,2:3]

    # Convert from polar to cartesian
    # Transforms only the non masked values from cylindrical to cartesian coordinates. Mask values are left unchanged.
    pxs =  pts * np.cos(phis)
    pys =  pts * np.sin(phis)
    pzs =  pts * np.sinh(etas)

    # Check to see if there are any infinities
    isinf = np.isinf(pzs)

    if np.any(isinf):
        print("Infinities in eta detected!")
        print(etas[isinf])
        raise ValueError("Infinity from sinh(eta) has been detected")

    # Returns the momentum vector in cartesian coordinates
    return np.concatenate([pxs, pys, pzs], axis=-1)


# In[108]:


def pad(x_values, maxsize, MASKVAL=-999):
    """
    Pads the inputs with nans to get to the maxsize
    """
    #Pad the non-regular arrays with null values until they are all of the same size. Then replace the nulls with MASVAL
    y_values = ak.fill_none(ak.pad_none(x_values, maxsize, axis=1, clip=True), MASKVAL)[:, :maxsize]
    return ak.to_regular(y_values, axis=1) #Return the regular arrays


# In[109]:


def flatten(x_values, maxsize=-1, MASKVAL=-999):
    """"Pads the input to ensure they are all of regular size and then zips together result"""
    y_values = {} 
    for field in x_values.fields:
        z_values = x_values[field]
        if maxsize > 0:
            z_values = pad(z_values, maxsize, MASKVAL)
        y_values[field] = z_values

    return ak.zip(y_values)


# In[110]:

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
    
    return K.mean(sum_loss)
   

def Mean_Squared_Error(true, meancovs_matrix):
    n_targets = np.shape(true)[1]
    means = meancovs_matrix[:, :n_targets]
    
    return K.mean(K.square(means-true), axis = -1)

# In[111]:

class PredictOnEpoch(tf.keras.callbacks.Callback):
    def __init__(self, model, x_test, y_test, model_name):
        self.model = model
        self.x_test = x_test
        self.y_test = y_test
        self.model_name = model_name

    def on_epoch_end(self, epoch, logs = {}):
        pred = self.model.predict(self.x_test)
        px_pred = pred[:,0]
        px_uncertainity = pred[:,3]
        true = self.y_test[:,0]

        global Predicted_Bhad_px
        global Predicted_Bhad_px_uncertainties
        
        if epoch > 1:
            Predicted_Bhad_px = np.concatenate((Predicted_Bhad_px,px_pred))
            Predicted_Bhad_px_uncertainties = np.concatenate((Predicted_Bhad_px_uncertainties, px_uncertainity))
        else:
            Predicted_Bhad_px = px_pred
            Predicted_Bhad_px_uncertainties = px_uncertainity

        lims = [-20,20]
        fig,ax = plt.subplots(figsize = (8,4))
        ax.scatter(true,px_pred, alpha = 0.6, color = '#32CD32', lw = 1, ec = "black")
        ax.plot(true, true, lw = 1, color = '#FF0000')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        fig.tight_layout()
        ax.set_title(f'Prediction Visualization Keras Callback - Epoch: {epoch}')
        fig.savefig(f'/home/physics/phujdj/DeepLearningParticlePhysics/Results/DeepNet_{self.model_name}-Epoch:{epoch}')
        plt.close()


def Normal_Accuracy_Metric(true,meanscovs_matrix):
    """
    The primary function of the LogNormal loss function is to determine
    best normal distribution to fit to the bhadron data. By including the 
    uncertainity however, the metric is not so usefull for error checking. 
    I have added accuracy metric to better measure the ability of the neural 
    network to predict the correct values
    """
    # Determine the number of features we are predicting
    n_targets = np.shape(true)[1]
    
    # Extract the means of the features
    means = meanscovs_matrix[:,:n_targets]

    Accuracy = []
    for n_target in range(n_targets):
        Accuracy.append(abs((means[:,n_target]-true[:,n_target])/true[:,n_target])*100)
    Accuracy = tf.convert_to_tensor(Accuracy)
    return keras.backend.mean(Accuracy)

# In[112]:


# In[113]:


def expontial_decay(lr0,s):
    def exponential_decay_fn(epoch):
        if epoch % 100 == 0:
            return lr0 * 10
        return lr0 * 0.40**(epoch/s)
    return exponential_decay_fn


# In[115]:


def DeepSetNeuralNetwork(track_layers, b_jet_layers, jet_layers, n_targets, n_features, optimizer, MASKVAL=-999):
    """
    This function lays out the Deep Set Neural Architecture
    - A neural network is applied first to the tracks to extract information from the tracks.
    - This information produces an ensemble space which, the outputs of which are then summed to produce
        the inputs for the next layer
    - A neural network is then applied to the jet data obtained from the tracks. 
        To perform current univariate regression.
    """
    #Input both the track and jet features.
    tracks = layers.Input(shape = (None, track_layers[0]))
    jets = layers.Input(shape = (n_features))

    #Mask out the the track and jet padding
    tracks = layers.Masking(mask_value=MASKVAL)(tracks)
    jets = layers.Masking(mask_value=MASKVAL)(jets)

    outputs_tracks = tracks
    outputs_jets = jets

    #counter = 0
    for nodes in track_layers[:-1]:
        #The first neural network is a series of dense layers and is applied to each track using the time distributed layer
        outputs_tracks = layers.TimeDistributed(layers.Dense(nodes,activation = "gelu",kernel_initializer= "he_normal"))(outputs_tracks)
        outputs_tracks = layers.BatchNormalization()(outputs_tracks) # Apply a batch norm to improve performance by preventing feature bias and overfitting

    outputs_tracks = layers.TimeDistributed(layers.Dense(track_layers[-1], activation = "softmax", kernel_initializer = "he_normal" ))(outputs_tracks) # Apply softmax to ouput the results of the track neural network as probabilities
    outputs_tracks = Sum()(outputs_tracks) # Sum the outputs to make use of permutation invariance

    outputs_jets = outputs_tracks
    #The b_jet network extracts information from the b_jet features.
    for nodes in b_jet_layers[:-1]:
        outputs_jets = layers.Dense(nodes, activation="gelu", kernel_initializer= "he_normal")(outputs_jets) # We use relu and the corresponding he_normal for the activation function and bias initializer

        outputs_jets = layers.LayerNormalization()(outputs_jets)
    outputs_jets = layers.Dense(b_jet_layers[-1], activation = "gelu", kernel_initializer="he_normal", kernel_regularizer=regularizers.l1_l2(0.01))(outputs_jets) # Apply softmax to ouput the results of the track neural network as probabilities

    outputs = layers.concatenate([outputs_tracks, outputs_jets], axis = -1)
    
    #counter = 0
    for nodes in jet_layers: #Repeat of the track neural network without the need for the timedistributed layers
        outputs = layers.Dense(nodes,activation = "gelu", kernel_initializer= "he_normal")(outputs)
        outputs = layers.BatchNormalization()(outputs)
    

    outputs = layers.Dense(n_targets + n_targets*(n_targets-1)//2)(outputs) # The output will have a number of neurons needed to form the mean covariance function of the loss function
    #outputs = layers.Dense(n_targets)(outputs)
    Model = keras.Model(inputs=[tracks, jets], outputs=outputs) #Create a keras model

    # Specify the neural network's optimizer and loss function
    Model.compile(
    optimizer=optimizer, # Optimizer used to train model
    loss= LogNormal_Loss_Function
    )

    return Model

