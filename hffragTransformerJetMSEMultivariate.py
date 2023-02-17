"""
*Filename: hffragTransformerMSEMultivariate
*Description: This python file lays out the neural network architeture for the transformer
*neural network architecture used to predict the mean, variances and covariances of the targets.
*Though originally designed to regress only the values of the targets, this has been extended to its variances and covariances
*through the addition of the multivariate negative likelihood full gaussian loss function.
*It builds on the original by adding more layers in the feedforward networks and changing the ouputs. It consists of an
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
from tensorflow_probability import distributions as tfd
from Sum import Sum

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

# Takes the means, variances and covariances predicted by the network and computes the multivariate negative likelihood loss
def Multivariate_Gaussian_Negative_Likelihood_Loss_Curve(true, mean_covariance_matrix):
    
    n_targets =  np.shape(true)[1]
    means = mean_covariance_matrix[:,:n_targets]
    uncertainties = tf.map_fn(lambda x: generate_triu(x, n_targets), mean_covariance_matrix, dtype = tf.float32)
    mvn = tfd.MultivariateNormalTriL(loc = means, scale_tril = uncertainties)
    log_likelihood = mvn.log_prob(true)
    return -log_likelihood

#Originally called the jet layer, the bhadsNet is inspired by the BhadsNet 
#jet network designed to regress the bhadron targets from the jet features
#It combines information obtained from encoder with the outputs 
#from a feed forward network progressing the b_jet features.
#Combining the b_jet and derived jet features the bhadsNet regress the bhadron targets.
def bhadsNet(n_features,d_model, n_targets, jet_layers):
        b_jets = layers.Input(shape = (n_features))
        jets_derived = layers.Input(shape = (d_model))
        jets = layers.concatenate([b_jets, jets_derived])
        output_bjets = jets
        output_bjets = layers.LayerNormalization()(output_bjets)
        
        counter = 0
        for nodes in jet_layers: #Repeat of the track neural network without the need for the timedistributed layers
            outputs = layers.Dense(nodes,activation = "gelu", kernel_initializer= "he_normal", kernel_regularizer = regularizers.l1_l2(1e-4))(output_bjets)
        if counter % 2 == 0:
            outputs = layers.Dropout(0.0001)(outputs)
        else:
            counter += 1
            outputs = layers.LayerNormalization()(outputs)

        outputs_target_jets = layers.Dense(n_targets, activation="linear")(outputs)
         
        outputs_variances_jets = layers.Dense(n_targets, activation='exponential')(outputs)

        outputs_covariances_jets = layers.Dense(n_targets*(n_targets-1)/2, activation='linear')(outputs)
        
        Outputs = layers.concatenate([outputs_target_jets,outputs_variances_jets, outputs_covariances_jets])

        Model = keras.Model(inputs = [b_jets,jets_derived], outputs = outputs_target_jets)
        
        return Model

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

# Calculates the mean squared errors for the predictions means
def Mean_Squared_Error(true, meancovs_matrix):
    n_targets = np.shape(true)[1]
    means = meancovs_matrix[:, :n_targets]
    return keras.backend.mean(keras.backend.square(means-true), axis = -1)

#The first step in the transformer architecture is to create an embedding of the track features. 
#This is done using a feedforward neural network consisting of several dense layers. The embedding
#is then passed onto the encoder. 
class ParticleEmbedder(tf.keras.layers.Layer):
    def __init__(self,d_model, MASKVAL):
        super().__init__()
        self.MASKVAL = MASKVAL
        self.d_model = d_model
        self.mask = tf.keras.layers.Masking(mask_value=MASKVAL)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.d_model,activation = "gelu", kernel_initializer = "he_normal")),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.d_model,activation = "gelu", kernel_initializer = "he_normal")),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.d_model,activation = "gelu", kernel_initializer = "he_normal")),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.d_model,activation = "gelu", kernel_initializer = "he_normal")),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.d_model, activation = "gelu", kernel_initializer = "he_normal")),
            tf.keras.layers.LayerNormalization()
        ])
        
    def call(self,x):
        outputs = self.mask(x)
        outputs = self.ffn(x)
        return outputs

#This transformer takes in b_jet features separately to the track features. The jet embedder is a feedforward network designed to turn the 
#b_jet features into an embedding that can be used by the BhadsNet layers.
class JetEmbedder(tf.keras.layers.Layer):
    def __init__(self,d_model, MASKVAL):
        super().__init__()
        self.MASKVAL = MASKVAL
        self.d_model = d_model
        self.mask = tf.keras.layers.Masking(mask_value=MASKVAL)
        self.embedding = tf.keras.Sequential([
            tf.keras.layers.Dense(d_model, activation="gelu",kernel_initializer="he_normal"),
            tf.keras.layers.Dense(d_model, activation="gelu",kernel_initializer="he_normal"),
            tf.keras.layers.Dense(d_model, activation="gelu",kernel_initializer="he_normal"),
            tf.keras.layers.Dense(d_model, activation="gelu",kernel_initializer="he_normal")
        ])
        
    def call(self,x):
        outputs = self.mask(x)
        outputs = self.embedding(outputs)
        return outputs

#Building on the bhadsNet the B_jet_layer is what is implemented into the transformer architecture to regress the b hadron targets.
class B_Jet_Layer(keras.layers.Layer):
    def __init__(self, n_targets, dff, jet_layers,  n_features,d_model, MASKVAL=-999):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.n_targets = n_targets
        self.dff = dff
        self.jet_layers = jet_layers
        self.JetNet = bhadsNet(self.n_features, self.d_model, self.n_targets,self.jet_layers)
    
    def call(self,x, context):
        x = self.JetNet([x, context])
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
# from the particle embedding returns derived jet features.
# It takes the jet features applies a jet embedder layer to turn the jet features into an jet embedding.
# The jet embedding and derived jet features used by the b jet layer to regress the b hadron.
#The encoder replaces the track network used in the DeepSets neural network architecture.
class Encoder(tf.keras.layers.Layer):
    def __init__(self, *, track_layers, d_model, num_heads, dff,MASKVAL = -999, dropout_rate = 0.1):
        super().__init__()

        self.num_track_layers = track_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.MASKVAL = MASKVAL
        self.dropout_rate = dropout_rate

        self.ParticleEmbedder = ParticleEmbedder(d_model=d_model,MASKVAL=MASKVAL)
        
        self.encoder_layers = [
            ParticleAttentionBlock(
            d_model=d_model,
            MASKVAL = MASKVAL,
            num_heads = num_heads,
            dff = dff,
            dropout_rate = dropout_rate
            )
        for _ in range(self.num_track_layers)]

        self.dropout = tf.keras.layers.Dropout(dropout_rate)
    
    def call(self, x):
        x = self.ParticleEmbedder(x)

        x = self.dropout(x)

        for increment in range(self.num_track_layers):
            x = self.encoder_layers[increment](x)

        return Sum()(x)

#Defines the transformer architecture used by the hffragTransformerJetMSEMultivariate.
#It consists of an encoder which transforms the track and jet features into derived features that the b-jet layers then uses to regress the bhadron
#features.
class hffragTransformer(tf.keras.Model):
    def __init__(self, *,track_layers, jet_layers, d_model, num_heads, dff, n_targets, n_features, MASKVAL = -999, dropout_rate = 0.1):
        super().__init__()

        self.encoder = Encoder(track_layers=track_layers, d_model=d_model, num_heads=num_heads, dff=dff, MASKVAL=MASKVAL, dropout_rate=dropout_rate)

        self.jet_layer = B_Jet_Layer(n_targets = n_targets, n_features=n_features, dff = dff, d_model= d_model, jet_layers = jet_layers, MASKVAL=MASKVAL)
        
    def call(self, inputs):
        Tracks, Jets = inputs

        Embedding = self.encoder(Tracks)

        Outputs = self.jet_layer(Jets,Embedding)

        return Outputs

