import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import awkward as ak
import uproot
import keras
import keras.layers as layers
from keras import regularizers
from Sum import Sum

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
    
    return sum_loss

class ParticleEmbbedder(tf.keras.layers.Layer):
    def __init__(self, track_layers,BATCH_SIZE, MASKVAL):
        super().__init__()
        self.track_layers = track_layers
        self.MASKVAL = MASKVAL
        self.BATCH_SIZE = BATCH_SIZE
        self.mask = tf.keras.layers.Masking(mask_value=MASKVAL)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(track_layers[0],activation = "elu")),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(track_layers[1],activation = "elu")),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(track_layers[2],activation = "elu")),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(track_layers[3],activation = "elu")),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.BATCH_SIZE, activation = "elu"))
        ])
        
    def call(self,x):
        outputs = self.mask(x)
        outputs = self.ffn(x)
        return outputs

class Jet_Layer(keras.layers.Layer):
    def __init__(self, dff, n_targets,d_model, MASKVAL=-999):
        super().__init__()
        self.mask = tf.keras.layers.Masking(mask_value=MASKVAL)
        self.n_targets = n_targets
        self.d_model = d_model
        self.dff = dff
        self.ffn = tf.keras.Sequential([
        layers.Dense(self.dff, activation = "elu"),
        layers.Dense(self.d_model),
        layers.Dense(self.n_targets+self.n_targets*(self.n_targets+1)//2),
        ])

    def call(self,x):
        x = self.mask(x)
        x = self.ffn(x)
        return x

class BaseAttention(tf.keras.layers.Layer):
    def __init__(self,MASKVAL, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()

class SelfAttention(BaseAttention):
    def call(self,x):
        attn_output = self.mha(
            query = x,
            value = x,
            key = x,
            kernel_regularizers = regularizers.l2(0.01))
        x  = self.add([x,attn_output])
        x = self.layernorm(x)
        return x

class FeedForward(tf.keras.layers.Layer):
    def __init__(self,d_model, dff, dropout = 0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(dff,activation = 'gelu'),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self,x):
        x = self.add([x,self.seq(x)])
        x = self.layer_norm(x)
        return x

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, MASKVAL = -999, dropout_rate = 0.1):
        super().__init__()
    
        self.self_attention = SelfAttention(
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

class Encoder(tf.keras.layers.Layer):
    def __init__(self, *, track_layers,num_layers, d_model, num_heads, dff,MASKVAL = -999, dropout_rate = 0.1):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.ParticleEmbbedder = ParticleEmbbedder(track_layers,d_model,MASKVAL)
        
        self.encoder_layers = [
            EncoderLayer(
            d_model=d_model,
            MASKVAL = MASKVAL,
            num_heads = num_heads,
            dff = dff,
            dropout_rate = dropout_rate
            )
        for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(dropout_rate)
    
    def call(self, x):
        x = self.ParticleEmbbedder(x)

        x = self.dropout(x)

        for increment in range(self.num_layers):
            x = self.encoder_layers[increment](x)

        x = Sum()(x)
        return x

class hffragTransformer(tf.keras.Model):
    def __init__(self, *,track_layers, num_layers, d_model, num_heads, dff, n_targets, MASKVAL = -999, dropout = 0.1):
        super().__init__()

        self.encoder = Encoder(track_layers = track_layers, d_model = d_model, num_layers= num_layers, num_heads=num_heads,dff = dff, MASKVAL= MASKVAL, dropout_rate = dropout)

        self.jet_layer = Jet_Layer(dff = dff, n_targets= n_targets, d_model = d_model)

    def call(self, inputs):
        x = inputs

        embedding = self.encoder(x)

        outputs = self.jet_layer(embedding)

        return outputs
    
    
