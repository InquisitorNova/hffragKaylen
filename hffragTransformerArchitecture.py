"""
*Filename: hffragTransformerArchitecture
*Description: This python file lays out the neural network architeture for the transformer
*neural network architecture used to predict the mean and variances of the targets.
*This is the original transformer architecture and the simplest. This transformer consists of an
*encoder consisting of several self attention layers followed by a feedforward neural network to take the features derived
*from the encoder to predict the targets. The file contains the loss functions and callbacks used by this transformer
*during training as well.
Date: 16/02/2023
Author: Kaylen Smith Darnbrook
"""

# Import relevant modules
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import awkward as ak
import uproot
import keras
import keras.layers as layers
from keras import regularizers
from Sum import Sum
from hffrag import fixedbinning
from hffrag import binneddensity

#An implementation of the gaussian negative likelihood loss function which returns the
#means and variances of the targets, assuming that targets are independent.
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
        sum_loss += (1/2)*keras.backend.log(2*np.pi) + logvariances[:,target] + ((true[:,target] - means[:,target])**2)/(2*(keras.backend.exp(logvariances[:,target])**2))
    
    return sum_loss

# Calculates the mean squared errors for the predictions means
def Mean_Squared_Error(true, meancovs_matrix):
    n_targets = np.shape(true)[1]
    means = meancovs_matrix[:, :n_targets]
    
    sum_loss = 0
    for target in range(n_targets):
        sum_loss += ((true[:,target]-means[:,target])**2)
    
    return sum_loss

# A callback returns a scatterplot of the predicted pxs and associated uncertainties per epoch
class PredictOnEpoch(tf.keras.callbacks.Callback):
    def __init__(self, model, x_test, y_test):
        self.model = model
        self.x_test = x_test
        self.y_test = y_test
    
    def on_epoch_end(self, epoch, logs = {}):
        pred = self.model.predict(self.x_test)
        px_pred = pred[0]
        px_uncertainity = pred[3]
        Figure = binneddensity(px_pred, fixedbinning(-300, 300,100),label = "Predicted x momentum [MeV] values")
        Figure.patch.set_facecolor('white')
        Figure.suptitle(f"Epoch {epoch}", fontsize = 20)
        Figure.savefig("/home/physics/phujdj/DeepLearningParticlePhysics/EpochPlots/PxPredictionOnEpoch-{Epoch}.png".format(Epoch = epoch),facecolor=Figure.get_facecolor())

        global Predicted_Bhad_px_means
        global Predicted_Bhad_px_uncertainties

        Predicted_Bhad_px_means = np.append(Predicted_Bhad_px_means,[epoch,np.mean(px_pred)])
        Predicted_Bhad_px_uncertainties = np.append(Predicted_Bhad_px_uncertainties, [epoch,np.mean(px_uncertainity)])

#The first step in the transformer architecture is to create an embedding of the track features. 
#This is done using a feedforward neural network consisting of several dense layers. The embedding
#is then passed onto the encoder. 
class ParticleEmbedder(tf.keras.layers.Layer):
    def __init__(self, track_layers,d_model, MASKVAL):
        super().__init__()
        self.track_layers = track_layers
        self.MASKVAL = MASKVAL
        self.d_model = d_model
        self.mask = tf.keras.layers.Masking(mask_value=MASKVAL)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(track_layers[0],activation = "tanh", kernel_initializer = "glorot_normal")),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(track_layers[1],activation = "tanh", kernel_initializer = "glorot_normal")),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(track_layers[2],activation = "tanh", kernel_initializer = "glorot_normal")),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(track_layers[3],activation = "tanh", kernel_initializer = "glorot_normal")),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.d_model, activation = "tanh", kernel_initializer = "glorot_normal"))
        ])
        
    def call(self,x):
        outputs = self.mask(x)
        outputs = self.ffn(x)
        return outputs

#The final step in transformer involves turning the derived features into predictions of the bhadron targets. 
#This is done by jet layer which is just a feed forward network.
class Jet_Layer(keras.layers.Layer):
    def __init__(self, dff, n_targets,d_model, MASKVAL=-999):
        super().__init__()
        self.n_targets = n_targets
        self.d_model = d_model
        self.dff = dff
        self.ffn = tf.keras.Sequential([
        layers.Dense(self.dff, activation = "tanh", kernel_initializer = "glorot_normal"),
        layers.Dense(self.d_model, activation = "tanh",  kernel_initializer = "glorot_normal"),
        layers.Dense(n_targets+n_targets*(n_targets+1)//2),
        ])

    def call(self,x):
        x = self.ffn(x)
        return x

#Defines the base attention layer used throughout the transformer.
class BaseAttention(tf.keras.layers.Layer):
    def __init__(self,MASKVAL, **kwargs):
        super().__init__()
        self.masking = tf.keras.layers.Masking(MASKVAL)
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()

#Defines the base self attention layer used in the encoder of the transformer.
class GlobalSelfAttention(BaseAttention):
    def call(self,x):
        attn_output = self.mha(
            query = x,
            value = x,
            key = x,
        )
        x = self.masking(x)
        x  = self.add([x,attn_output])
        x = self.layernorm(x)
        return x

#Defines the feedforward layers used between the self attention layers in the encoder.
class FeedForward(tf.keras.layers.Layer):
    def __init__(self,d_model, dff, dropout = 0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(dff,activation = "gelu",  kernel_initializer = "he_normal"),
            tf.keras.layers.Dense(d_model, activation = "gelu",  kernel_initializer = "he_normal"),
            tf.keras.layers.Dropout(dropout)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self,x):
        x = self.add([x,self.seq(x)])
        x = self.layer_norm(x)
        return x

#Defines the attention block which takes the particle embeddings and 
#extracts the cross patterns between the tracks and their features.
class ParticleAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, MASKVAL = -999, dropout_rate = 0.1):
        super().__init__()
    
        self.self_attention = GlobalSelfAttention(
            MASKVAL = MASKVAL,
            num_heads = num_heads,
            key_dim = d_model,
            dropout = dropout_rate
        )
    
        self.ffn = FeedForward(d_model, dff)

    def call(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x

#The encoder takes the track features, applies a particle embedder layer to turn the tracks features into an embedding and then
# from the particle embedding returns derived features used by the jet layer to regress the b hadron.
#The encoder replaces the track network used in the DeepSets neural network architecture.
class Encoder(tf.keras.layers.Layer):
    def __init__(self, *, track_layers,num_layers, d_model, num_heads, dff,MASKVAL = -999, dropout_rate = 0.1):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.track_layers = track_layers
        self.d_model = d_model

        self.ParticleEmbedder = ParticleEmbedder(track_layers=track_layers,d_model=d_model,MASKVAL=MASKVAL)
        
        self.encoder_layers = [
            ParticleAttentionBlock(
            d_model=d_model,
            MASKVAL = MASKVAL,
            num_heads = num_heads,
            dff = dff,
            dropout_rate = dropout_rate
            )
        for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(dropout_rate)
    
    def call(self, x):
        x = self.ParticleEmbedder(x)

        x = self.dropout(x)

        for increment in range(self.num_layers):
            x = self.encoder_layers[increment](x)

        return Sum()(x)

#Defines the transformer architecture used by the hffragTransformer.
#It consists of an encoder which transforms the track features into derived features that the jet layers then uses to regress the bhadron
#features.
class hffragTransformer(tf.keras.Model):
    def __init__(self, *,track_layers, num_layers, d_model, num_heads, dff, n_targets, MASKVAL, dropout = 0.1):
        super().__init__()

        self.encoder = Encoder(track_layers = track_layers, d_model = d_model, num_layers= num_layers, num_heads=num_heads,dff = dff, MASKVAL= MASKVAL, dropout_rate = dropout)

        self.jet_layer = Jet_Layer(n_targets= n_targets, dff = dff, d_model= d_model)
        
    def call(self, inputs):

        context = self.encoder(inputs)

        outputs = self.jet_layer(context)

        return outputs


