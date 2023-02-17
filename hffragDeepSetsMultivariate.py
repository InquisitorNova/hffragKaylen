"""
*Filename: hffragDeepSetsMultivariate
*Description: This python file lays out the neural network architecture for the
* DeepSets neural network used to predict the mean, variances and covariances of
* the targets. The python file also includes the definition for the loss the 
* neural network uses during training and its callbacks. This DeepSets is a variation
* of the the original deepsets introduced by the supervisor in that it regress the 
* mean and uncertainties seperately.
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

# Inspired by the original DeepSets paper and the paper which introduced the negative likelihood loss, 
# this version of deepsets predicts the mean and associated uncertainties using two branches of layers 
# which combine there output at the end.
def hffragDeepSets(track_layers, jet_layers,b_jet_layers, n_targets, regularizer_strength, n_features, optimizer, loss_curve = Multivariate_Gaussian_Negative_Likelihood_Loss_Curve, NumDropout = 0, Dropout_rate = 0.001, MASKVAL=-999):
        tracks = layers.Input(shape = (None,track_layers[0]))
        jets = layers.Input(shape = (n_features))

        output_bjets = jets
        output_bjets = layers.Masking(mask_value=MASKVAL)(output_bjets)
        output_bjets = layers.BatchNormalization()(output_bjets)
        
        for jet_layer in b_jet_layers:
            output_bjets = layers.Dense(jet_layer, activation='gelu', kernel_initializer= "he_normal")(output_bjets)
            output_bjets = layers.BatchNormalization()(output_bjets)

        output_jets= layers.Dense(b_jet_layers[-1], activation = "softmax")(output_bjets)

        outputs_tracks_Target = tracks  # Creates another layer to pass the inputs onto the ouputs
        outputs_tracks_Target = layers.Masking(mask_value=MASKVAL)(outputs_tracks_Target) # Masks the MASKVAl values
        outputs_tracks_Target = layers.BatchNormalization()(outputs_tracks_Target)
        
        counter = NumDropout
        for nodes in track_layers[:-1]:
            #The first neural network is a series of dense layers and is applied to each track using the time distributed layer
            outputs_tracks_Target = layers.TimeDistributed( 
            layers.Dense(nodes, activation="gelu", kernel_initializer= "he_normal", kernel_regularizer = regularizers.l1_l2(regularizer_strength)))(outputs_tracks_Target) # We use relu and the corresponding he_normal for the activation function and bias initializer
            if  counter > 0:
                outputs_tracks_Target = layers.Dropout(Dropout_rate)(outputs_tracks_Target)
            else:
                counter -= 1

            outputs_tracks_Target = layers.LayerNormalization()(outputs_tracks_Target) # Apply a batch norm to improve performance by preventing feature bias and overfitting

        outputs_tracks_Target = layers.TimeDistributed(layers.Dense( 
            track_layers[-1], activation = "softmax"))(outputs_tracks_Target) # Apply softmax to ouput the results of the track neural network as probabilities
        outputs_tracks_Target = Sum()(outputs_tracks_Target) # Sum the outputs to make use of permutation invariance

        outputs_Target_jets = layers.concatenate([outputs_tracks_Target,output_jets])
        
        counter = NumDropout
        for nodes in jet_layers: #Repeat of the track neural network without the need for the timedistributed layers
            outputs_Target_jets = layers.Dense(nodes, activation='gelu', kernel_initializer= "he_normal", kernel_regularizer = regularizers.l1_l2(regularizer_strength))(outputs_Target_jets)
        
            if  counter > 0:
                outputs_Target_jets = layers.Dropout(Dropout_rate)(outputs_Target_jets)
            else:
                counter -= 1
        
            outputs_Target_jets= layers.LayerNormalization()(outputs_Target_jets)
        
        outputs_Target_jets = layers.Dense(n_targets, activation='linear')(outputs_Target_jets)
        
        outputs_tracks_Uncertainties = tracks  # Creates another layer to pass the inputs onto the ouputs
        outputs_tracks_Uncertainties = layers.Masking(mask_value=MASKVAL)(outputs_tracks_Uncertainties) # Masks the MASKVAl values
        outputs_tracks_Uncertainties = layers.BatchNormalization()(outputs_tracks_Uncertainties)
        
        counter = NumDropout
        for nodes in track_layers[:-1]:
            #The first neural network is a series of dense layers and is applied to each track using the time distributed layer
            outputs_tracks_Uncertainties = layers.TimeDistributed( 
            layers.Dense(nodes, activation="gelu", kernel_initializer= "he_normal", kernel_regularizer = regularizers.l1_l2(regularizer_strength)))(outputs_tracks_Uncertainties) # We use relu and the corresponding he_normal for the activation function and bias initializer
            if  counter > 0:
                outputs_tracks_Uncertainties = layers.Dropout(Dropout_rate)(outputs_tracks_Uncertainties)
            else:
                counter -= 1

            outputs_tracks_Uncertainties = layers.LayerNormalization()(outputs_tracks_Uncertainties) # Apply a batch norm to improve performance by preventing feature bias and overfitting

        outputs_tracks_Uncertainties = layers.TimeDistributed(layers.Dense( 
            track_layers[-1], activation = "softmax"))(outputs_tracks_Uncertainties) # Apply softmax to ouput the results of the track neural network as probabilities
        outputs_tracks_Uncertainties = Sum()(outputs_tracks_Uncertainties) # Sum the outputs to make use of permutation invariance

        outputs_Uncertainties_jets = layers.concatenate([outputs_tracks_Uncertainties,output_jets])

        counter = NumDropout
        for nodes in jet_layers: #Repeat of the track neural network without the need for the timedistributed layers
            outputs_Uncertainties_jets = layers.Dense(nodes, activation='gelu', kernel_initializer= "he_normal", kernel_regularizer = regularizers.l1_l2(regularizer_strength))(outputs_Uncertainties_jets)
        
            if  counter > 0:
                outputs_Uncertainties_jets = layers.Dropout(Dropout_rate)(outputs_Uncertainties_jets)
            else:
                counter -= 1
        
            outputs_Uncertainties_jets= layers.LayerNormalization()(outputs_Uncertainties_jets)


        #Combines the outputs from the two branches of layers to predict the mean, covariance and uncertainties. 
        outputs_variances_jets = layers.Dense(n_targets, activation='exponential')(outputs_Uncertainties_jets)

        outputs_covariances_jets = layers.Dense(n_targets*(n_targets-1)/2, activation='linear')(outputs_Uncertainties_jets)
        
        Outputs = layers.concatenate([outputs_Target_jets,outputs_variances_jets, outputs_covariances_jets])
    
        Model = keras.Model(inputs = [tracks,jets], outputs = outputs_Target_jets)
        
        Model.compile(
            optimizer = optimizer,
            loss = loss_curve,
            jit_compile = True
        )
        return Model