#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import relevant modules
import os
import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras.layers as layers
from keras import callbacks
import keras
import uproot
from Sum import Sum
import sklearn as sk
from numpy.lib.recfunctions import structured_to_unstructured
from sklearn.model_selection import train_test_split
import pandas as pd
import hffrag
import keras_tuner as kt
from hffrag import fixedbinning
from hffrag import binneddensity


# In[2]:


#A magic operator to allow Jupyter Notebooks to display matplotlib plots as outputs of cells
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# The data is being stored in a tree datastructure.
# We access the charm root using this command
tree = uproot.open("hffrag.root:CharmAnalysis")


# In[4]:


# Initial parameters
MASKVAL = -999 # This value is introduced to ensure arrays are regular (Of the same size). They will be masked later by the network
MAXTRACKS = 512 # This value is the maximum number of tracks allowed per event
BATCHSIZE = 32 # This is the batch size of the mini batches used during training
EPOCHS = 1000 # This is the default number of epochs for which the neural network will train providing that early stopping does not occur
MAXEVENTS = 1e24 #This is the maximum number of events that will the program will accept
LR = 5e-3 #This is the default learning rate


# In[5]:


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


# In[6]:


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


# In[7]:


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


# In[8]:


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


# In[9]:


def pad(x_values, maxsize, MASKVAL=-999):
    """
    Pads the inputs with nans to get to the maxsize
    """
    #Pad the non-regular arrays with null values until they are all of the same size. Then replace the nulls with MASVAL
    y_values = ak.fill_none(ak.pad_none(x_values, maxsize, axis=1, clip=True), MASKVAL)[:, :maxsize]
    return ak.to_regular(y_values, axis=1) #Return the regular arrays


# In[10]:


def flatten(x_values, maxsize=-1, MASKVAL=-999):
    """"Pads the input to ensure they are all of regular size and then zips together result"""
    y_values = {} 
    for field in x_values.fields:
        z_values = x_values[field]
        if maxsize > 0:
            z_values = pad(z_values, maxsize, MASKVAL)
        y_values[field] = z_values

    return ak.zip(y_values)


# In[11]:


def LogNormal_Loss_Function(true, meanscovs_matrix):
    """
    This is a loss function hand crafted for the task of ensuring the neural network 
    learns to predict the true value of the transverse momentum and it's uncertainty
    The logNormal constrains the neural network, by forcing upon it what it's output layers should be
    and what the weights and biases of the neural network will be in order to predict the means, variances and covariances
    """
    n_targets = np.shape(true)[1]
    # The first n_target of the features are the means
    means = meanscovs_matrix[:, :n_targets]
    # The second n_target of the feautres are the standard deviations
    logsigma = meanscovs_matrix[:, n_targets:2*n_targets]
    # The rest of the targets are the covariances
    logcosigma = meanscovs_matrix[:,2*n_targets:]

    loss = 0
    for n_target in range(n_targets): #Sum the individual losses and use that as the loss for the neural network
        loss += ((means[:, n_target] - true[:, n_target])**2) / (2 * keras.backend.exp(logsigma[:, n_target])**2) + logsigma[:, n_target]

    # Build loss function
    return loss


# In[12]:


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


# In[13]:


def LogNormal_Loss_Function_Check(true,meanscovs_matrix):
    """The role of this function is to calculate the loss for each individual b jet. This is used for the purpose of error checking"""
    n_targets = np.shape(true)[0]
    # Obtain data from convarience matrix
    means = meanscovs_matrix[0, :n_targets]
    # ensure diagonal is postive:
    logsigma = meanscovs_matrix[0, n_targets:2*n_targets]

    loss = []
    for n_target in range(n_targets):
        loss.append(((means[n_target] - true[n_target])**2) / (2 * keras.backend.exp(logsigma[n_target])**2) + logsigma[n_target])
    return loss


# In[14]:


def expontial_decay(lr0,s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.95**(epoch/s)
    return exponential_decay_fn
exponential_decay_fn = expontial_decay(lr0 = LR,s = 30)


# In[15]:


from keras.utils.vis_utils import plot_model


# In[16]:


def DeepSetNeuralNetwork(track_layers, jet_layers, n_targets,Learning_rate, MASKVAL=-999):
    """
    This function lays out the Deep Set Neural Architecture
    - A neural network is applied first to the tracks to extract information from the tracks.
    - This information produces an ensemble space which, the outputs of which are then summed to produce
        the inputs for the next layer
    - A neural network is then applied to the jet data obtained from the tracks. 
        To perform current univariate regression.
    """
    inputs = layers.Input(shape=(None, track_layers[0])) # Creates a layer for each input
    outputs = inputs  # Creates another layer to pass the inputs onto the ouputs
    outputs = layers.Masking(mask_value=MASKVAL)(outputs) # Masks the MASKVAl values

    counter = 0
    for nodes in track_layers[:-1]:
        #The first neural network is a series of dense layers and is applied to each track using the time distributed layer
        outputs = layers.TimeDistributed( 
            layers.Dense(nodes, activation="elu", kernel_initializer= "he_normal",kernel_regularizer = keras.regularizers.l2(0.01)))(outputs) # We use relu and the corresponding he_normal for the activation function and bias initializer
        if counter % 2 == 0: # Every two layers apply a dropout
            outputs = layers.Dropout(0.2)(outputs)
        else:
            counter += 1
        outputs = layers.BatchNormalization()(outputs) #Apply a batch norm to improve performance by preventing feature bias and overfitting

    outputs = layers.TimeDistributed(layers.Dense( 
        track_layers[-1], activation='softmax'))(outputs) # Apply softmax to ouput the results of the track neural network as probabilities
    outputs = Sum()(outputs) # Sum the outputs to make use of permutation invariance

    counter = 0
    for nodes in jet_layers: #Repeat of the track neural network without the need for the timedistributed layers
        outputs = layers.Dense(nodes, activation='elu', kernel_initializer= "he_normal",kernel_regularizer = keras.regularizers.l2(0.01))(outputs)
        if counter % 2 == 0:
            outputs = layers.Dropout(0.2)(outputs)
        else:
            counter += 1
        outputs = layers.BatchNormalization()(outputs)

    outputs = layers.Dense(n_targets+n_targets*(n_targets+1)//2)(outputs) # The output will have a number of neurons needed to form the mean covariance function of the loss func

    Model = keras.Model(inputs=inputs, outputs=outputs) #Create a keras model

    # Specify the neural network's optimizer and loss function
    Model.compile(
    optimizer=keras.optimizers.Nadam(learning_rate=Learning_rate,clipnorm = 1.0), # Optimizer used to train model
    metrics = [Normal_Accuracy_Metric], # Metric used to assess true performance of model
    loss=LogNormal_Loss_Function, #Loss function
    )

    return Model


# In[17]:


# Select the features we wish to study
track_features = ["AnalysisTracks_pt", "AnalysisTracks_eta", "AnalysisTracks_phi", "AnalysisTracks_z0sinTheta",
                  "AnalysisTracks_d0sig", "AnalysisTracks_d0", "AnalysisTracks_d0sigPV", "AnalysisTracks_d0PV"]
jet_features = ["AnalysisAntiKt4TruthJets_pt", "AnalysisAntiKt4TruthJets_eta", "AnalysisAntiKt4TruthJets_phi",
                "AnalysisAntiKt4TruthJets_ghostB_pt", "AnalysisAntiKt4TruthJets_ghostB_eta","AnalysisAntiKt4TruthJets_ghostB_phi"]


# In[18]:


# Read in the data from the root file
features = tree.arrays(jet_features+track_features, entry_stop=MAXEVENTS)


# In[19]:


# Select the events of interest
events = features[ak.sum(
    features["AnalysisAntiKt4TruthJets_pt"] > 25000, axis=1) > 0]


# In[20]:


# Displays the number of jets being trained on
jets = events[jet_features][:, 0]
print("The number of jets to train on is: ", len(jets))
print("The number of track features is: ",len(track_features))


# In[21]:


# Select tracks from the events
tracks = events[track_features]

# Match the tracks to the jets
matchedtracks = tracks[Match_Tracks(jets, tracks)]

# Pad and Flatten the data
matchedtracks = flatten(matchedtracks, MAXTRACKS)


# In[22]:


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


# In[23]:


print(np.shape(bhads)) #Check the shape of the neural network
print(np.shape(jet_features[:-1])) #Check for shape of the jet features
print(jets[jet_features[0]]) # Check the jets


# In[24]:


# Transform the jet and tracks to unstructed data.
jets = structured_to_unstructured(jets[jet_features[:-3]])
matchedtracks = structured_to_unstructured(matchedtracks)
print(np.shape(jets))


# In[25]:


#Check the matchtracks are the correct shape
print(matchedtracks[:, 0:1])
print(np.shape(matchedtracks[:, :, 3]))


# In[26]:


# Convert the coordinates of the b jets and tracks to cartesian coordinates
tracks_p = pt_eta_phi_2_px_py_pz_tracks(matchedtracks.to_numpy())
bhads = pt_eta_phi_2_px_py_pz_jets(bhads)

#Check the shape of the momenta of the tracks and the rest of the data is consistent
print(np.shape(tracks_p))
print(np.shape(matchedtracks[:, :, 3:]))

#Combine the momenta of the tracks with the rest of the track features to form the track dataset
tracks = np.concatenate([tracks_p,matchedtracks[:,:,3:].to_numpy()],axis = 2)


# In[27]:


#Check that this is all the correct shape
print(np.shape(tracks))
print(np.shape(bhads))
print(tracks[0,0])
print(bhads[0])


# In[28]:


# Builds the deep neural network
track_layers = [32,32,32,32,32]
jet_layers = [64,64,64,64,64]
DeepNet = DeepSetNeuralNetwork(
    [len(track_features)]+track_layers, jet_layers,3, LR)


# In[29]:


#Summarises the Neural Network Architecture
DeepNet.summary()


# In[30]:


plot_model(DeepNet, to_file ="NetworkArchitecture.png", show_shapes = True, show_layer_names = True)


# In[31]:


# Split the data into training and validation sets.
X_train, X_valid, y_train, y_valid = train_test_split(
    tracks, bhads, train_size=0.7, random_state=42)
#Save the training and validation datasets.
np.save("/home/physics/phujdj/DeepLearningParticlePhysics/TrainingAndValidationData/X_train_data.npy",X_train)
np.save("/home/physics/phujdj/DeepLearningParticlePhysics/TrainingAndValidationData/X_valid_data.npy",X_valid)
np.save("/home/physics/phujdj/DeepLearningParticlePhysics/TrainingAndValidationData/y_train_data.npy",y_train)
np.save("/home/physics/phujdj/DeepLearningParticlePhysics/TrainingAndValidationData/y_valid_data.npy",y_valid)


# In[32]:


#Check for the of the training and validation sets
print(np.shape(X_train), np.shape(y_train))


# In[33]:


print(np.min(X_train[0,:,2]))
print(np.max(X_train[0,:,2]))


# In[34]:


print(np.min(y_train[0,0]))
print(np.max(y_train[0,0]))


# In[35]:


print(np.shape(tracks[0]))


# In[36]:


# Introduce early_stopping to prevent overfitting
early_stopping = callbacks.EarlyStopping(
    min_delta=0.001,  # The minimum amount of change to count as an improvement
    patience=15,  # The number of epochs to wait before stopping
    restore_best_weights=True,  # Keep the best weights
)
# Prevent spikes in the validation and training loss due to the gradient descent kicking the network out of a local minima
reduce_learn_on_plateau = callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.8, patience=20, min_lr=1e-6)

# Save the weights of the model to allow reuse in future.
path = "/home/physics/phujdj/DeepLearningParticlePhysics/CheckPoints/DeepNetWeights&Biases.ckpt"
checkpoint_dir = os.path.dirname(path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=path,
                                                 save_weights_only=True, verbose=0, save_best_only=True)

# Learning Scheduler:
learning_scheduler = tf.keras.callbacks.LearningRateScheduler(exponential_decay_fn)


# In[37]:


# Train the neural network
history = DeepNet.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=BATCHSIZE,
    epochs=EPOCHS,
    callbacks=[cp_callback,learning_scheduler]  # Enter call back
)


# In[38]:


# Plot the loss and validation curves vs epoch
history_df = pd.DataFrame(history.history)
np.log(history_df.loc[:, ["loss", "val_loss"]]).plot()


# In[ ]:


# Output to the console the minimum epoch
print("Minimum validation loss: {}".format(history_df["val_loss"].min()))


# In[ ]:


#Predict the momentas for a single jet and determine the loss
print(np.shape(tracks[0]))
print(DeepNet.predict(tracks)[0])
print(bhads[0])


# In[ ]:


# Calculate the individual loss for each feature
LogNormal_Loss_Function_Check(bhads[0],DeepNet.predict(tracks[0]))


# In[ ]:


#Evaluate the entire performance of the model
loss = DeepNet.evaluate(tracks,bhads,verbose = 2)
print("The Loaded DeepNet has loss: ", loss)


# In[ ]:


from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor


# In[ ]:


X_train = np.ma.masked_equal(X_train, -999)
Regressor = linear_model.LinearRegression()
print(np.shape(X_train[:,:,0]))
Regressor.fit(X_train[:,:,0],y_train[:,0])


# In[ ]:


print(np.shape(tracks[:,:,0]))
X_1 = np.ma.masked_equal(tracks[:,:,0],-999)
Regressor.score(X_1,bhads[:,0])


# In[ ]:


ForestRegressor = RandomForestRegressor(n_estimators = 200, max_depth = 14, random_state = 42)
ForestRegressor.fit(X_train[:,:,0],y_train[:,0])


# In[ ]:


ForestRegressor.score(X_1,bhads[:,0])


# In[ ]:


Predictions = ForestRegressor.predict(X_1)
Predictions[50]


# In[ ]:


binneddensity(Predictions - bhads[:,0], fixedbinning(-100000,100000,100), xlabel ="RandomForestModel")


# In[ ]:


binneddensity(Predictions, fixedbinning(-100000,100000,100), xlabel ="RandomForestModel")


# In[ ]:


from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler


# In[ ]:


Regressor = SGDRegressor(loss = "squared_error")
scaler = StandardScaler()
scaler.fit_transform(X_train[:,:,0],y_train[:,0])
Regressor.fit(X_train[:,:,0],y_train[:,0])


# In[ ]:


Regressor.score(X_1,bhads[:,0])

