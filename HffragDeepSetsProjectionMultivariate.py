"""
*Filename: hffragDeepSetsProjection
*Description: This python file lays out the neural network architecture for the
* DeepSets neural network used to predict the mean, variances and covariances of
* the targets. The python file also includes the definition for the loss the 
* neural network uses during training and its callbacks. This DeepSets is a variation
* of the the original deepsets introduced by the supervisor in that it makes use of 
* monte carlo dropout, skip connections and also classifies the bhadrons by pdgids.
Date: 16/02/2023
Author: Kaylen Smith Darnbrook
"""

# Import relevant modules
import awkward as ak
import numpy as np
import tensorflow as tf
import keras.layers as layers
from keras import regularizers
import keras.backend as K
import keras
from Sum import Sum
from tensorflow_probability import distributions as tfd
import matplotlib.pyplot as plt

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

# Overides the dropout layer so that dropout remains on after training.
class MCDropout(keras.layers.Dropout):
    def call(self, inputs):
        return super().call(inputs, training = True)

#Creates a custom layer for the time distributed layers which makes use of skip connections to speed up training and improve performance.
class TimeDistributedResidualUnits(keras.layers.Layer):
    def __init__(self, d_model, regularizer_strength = 0.01, dropout = 0.001):
        super().__init__()
        self.seq = tf.keras.Sequential([
            layers.TimeDistributed(layers.Dense(d_model, activation = "gelu",  kernel_initializer = "he_normal", kernel_regularizer= regularizers.l1_l2(regularizer_strength))),
            layers.TimeDistributed(layers.Dense(d_model, activation = "gelu",  kernel_initializer = "he_normal", kernel_regularizer= regularizers.l1_l2(regularizer_strength))),
            MCDropout(dropout)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x


#Creates a custom layer for the dense layers which makes use of skip connections to speed up training and improve performance.
class ResidualUnits(keras.layers.Layer):
    def __init__(self, d_model, regularizer_strength = 0.01, dropout = 0.001):
        super().__init__()
        self.seq = tf.keras.Sequential([
            layers.Dense(d_model, activation = "gelu",  kernel_initializer = "he_normal", kernel_regularizer= regularizers.l1_l2(regularizer_strength)),
            layers.Dense(d_model, activation = "gelu",  kernel_initializer = "he_normal", kernel_regularizer= regularizers.l1_l2(regularizer_strength)),
            MCDropout(dropout)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x

# Generates an upper triangular matrix from the variances and covariances of the predictions
def generate_triu(mean_covariance_matrix, n_targets):
    variances = mean_covariance_matrix[n_targets:2*n_targets]
    covariances = mean_covariance_matrix[2*n_targets:]

    ones = tf.ones((n_targets,n_targets), dtype = tf.float32)
    mask_a = tf.linalg.band_part(ones, 0, -1)
    mask_b = tf.linalg.band_part(ones, 0, 0)
    mask = tf.subtract(mask_a,mask_b)
    zero = tf.constant(0, dtype = tf.float32)
    non_zero = tf.not_equal(mask,zero)
    indices = tf.where(non_zero)
    covariances = tf.sparse.SparseTensor(indices, covariances, dense_shape = tf.cast((n_targets,n_targets), dtype = tf.int64))
    covariances = tf.sparse.to_dense(covariances)
    variances = tf.linalg.diag(variances)
    uncertainties = variances + covariances

    return uncertainties


# Takes the means, variances and covariances predicted by the network and computs the multivariate negative likelihood loss
def Multivariate_Gaussian_Negative_Likelihood_Loss_Curve(true, mean_covariance_matrix):
    
    n_targets =  np.shape(true)[1]
    means = mean_covariance_matrix[:,:n_targets]
    uncertainties = tf.map_fn(lambda x: generate_triu(x, n_targets), mean_covariance_matrix, dtype = tf.float32)
    mvn = tfd.MultivariateNormalTriL(loc = means, scale_tril = uncertainties)
    log_likelihood = mvn.log_prob(true)
    return -log_likelihood


# Calculates the mean squared errors for the predictions means
def Mean_Squared_Error(true, meancovs_matrix):
    n_targets = np.shape(true)[1]
    means = meancovs_matrix[:, :n_targets]
    
    return K.mean(K.square(means-true), axis = -1)


# A callback returns a scatterplot of the predicted pxs and associated uncertainties per epoch
class PredictOnEpoch(tf.keras.callbacks.Callback):
    def __init__(self, model, x_test, y_test, Bjets, model_name):
        self.model = model
        self.x_test = x_test
        self.y_test = y_test
        self.Bjets = Bjets
        self.model_name = model_name

    def on_epoch_end(self, epoch, logs = {}):
        pred = self.model.predict(self.x_test)
        px_pred = pred[:,0] * self.Bjets[:,0]
        px_uncertainity = pred[:,3] * self.Bjets[:,0]
        true = self.y_test[:,0]

        global Predicted_Bhad_px
        global Predicted_Bhad_px_uncertainties
        
        if epoch > 1:
            Predicted_Bhad_px = np.concatenate((Predicted_Bhad_px,px_pred))
            Predicted_Bhad_px_uncertainties = np.concatenate((Predicted_Bhad_px_uncertainties, px_uncertainity))
        else:
            Predicted_Bhad_px = px_pred
            Predicted_Bhad_px_uncertainties = px_uncertainity

        lims = [-400000, 800000]
        fig,ax = plt.subplots(figsize = (8,4))
        ax.scatter(true,px_pred, alpha = 0.6, color = '#32CD32', lw = 1, ec = "black")
        ax.plot(true, true, lw = 1, color = '#FF0000')
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        fig.tight_layout()
        ax.set_title(f'Prediction Visualization Keras Callback - Epoch: {epoch}')
        fig.savefig(fr'C:\Users\44730\hffragKaylen\hffragKaylen\Results\{self.model_name}-Epoch:{epoch}')
        plt.close()

#Called DeepSetsProjection due to its original purpose being to 
#improve the performance of the predictions of the projected momenta,
#DeepSetsProjection makes use of the original DeepSets architecture with monte carlo dropout and skip connections
#to regress the bhadron features and classify the bhadrons.
def DeepSetsProjection(track_layers, jet_layers, b_jet_layers, n_targets, n_targets_classification, regularizer_strength, n_features, Dropout_rate = 0.0001):
    
    #Input both the track and jet features.
    tracks = layers.Input(shape = (None, track_layers[0]))
    jets = layers.Input(shape = (n_features))
    
    #Mask out the the track and jet padding
    tracks = layers.Masking(mask_value=MASKVAL)(tracks)
    jets = layers.Masking(mask_value=MASKVAL)(jets)

    outputs_tracks = tracks
    outputs_jets = jets
    
    #The track network extracts information from the track features.
    for nodes in track_layers[:-1]:
        outputs_tracks = layers.Dense(nodes, activation = "gelu", kernel_initializer="he_normal")(outputs_tracks)
        outputs_tracks = TimeDistributedResidualUnits(nodes,regularizer_strength, Dropout_rate)(outputs_tracks)

    outputs_tracks = layers.TimeDistributed(layers.Dense( 
    track_layers[-1], activation = "gelu"))(outputs_tracks) # Apply softmax to ouput the results of the track neural network as probabilities
    outputs_tracks = Sum()(outputs_tracks) # Sum the outputs to make use of permutation invariance
    
    #The b_jet network extracts information from the b_jet features.
    for nodes in b_jet_layers[:-1]:
        outputs_jets = layers.Dense(nodes, activation="gelu", kernel_initializer= "he_normal")(outputs_jets) # We use relu and the corresponding he_normal for the activation function and bias initializer

        outputs_jets = layers.LayerNormalization()(outputs_jets)
    outputs_jets = layers.Dense(b_jet_layers[-1], activation = "gelu", kernel_initializer="he_normal", kernel_regularizer=regularizers.l1_l2(regularizer_strength))(outputs_jets) # Apply softmax to ouput the results of the track neural network as probabilities

    outputs = layers.concatenate([outputs_tracks, outputs_jets], axis = -1)

    #The jet network uses information obtained from the track and b_jet networks to regress the bhadron targets and classify the bhadrons
    for nodes in jet_layers[:-1]:
        outputs = layers.Dense(nodes, activation = "gelu", kernel_initializer="he_normal", kernel_regularizer= regularizers.l1_l2(regularizer_strength))(outputs)
        outputs = ResidualUnits(nodes,regularizer_strength, Dropout_rate)(outputs)

    #The outputs for regressing the bhadron targets.
    outputs_Target_jets = layers.Dense(n_targets, activation='linear', name = "HuberLoss")(outputs)
    outputs_variances_jets = layers.Dense(n_targets, activation='exponential')(outputs)
    outputs_covariances_jets = layers.Dense(n_targets*(n_targets-1)/2, activation='linear')(outputs)

    #The outputs for classifying the remaining bhadron targets.
    outputs_Target_jets_mass = layers.Dense(256, activation='gelu', kernel_initializer= "he_normal")(outputs_Target_jets)
    outputs_Target_jets_mass = layers.Dense(256, activation='gelu', kernel_initializer= "he_normal")(outputs_Target_jets_mass)
    outputs_Target_jets_mass = layers.Dense(n_targets_classification, activation='softmax', name = "MassOutput")(outputs_Target_jets_mass)

    Outputs_Multivariate = layers.concatenate([outputs_Target_jets,outputs_variances_jets, outputs_covariances_jets], name = "MultivariateLoss")
    
    Model = keras.Model(inputs = [tracks,jets], outputs = [Outputs_Multivariate, outputs_Target_jets_mass])

    return Model

