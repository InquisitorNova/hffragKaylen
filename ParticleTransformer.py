import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.layers as layers
import keras 
from Sum import Sum

# Initial parameters
MASKVAL = -999 # This value is introduced to ensure arrays are regular (Of the same size). They will be masked later by the network
MAXTRACKS = 32 # This value is the maximum number of tracks allowed per event
BATCHSIZE = 64 # This is the batch size of the mini batches used during training
EPOCHS = 100 # This is the default number of epochs for which the neural network will train providing that early stopping does not occur
MAXEVENTS = 1e20 #This is the maximum number of events that will the program will accept
LR = 1e-4 #This is the default learning rate

class ParticleMasker(tf.keras.layers.Layer):
    def __init__(self,MASKVAL):
        super().__init__()
        self.mask = tf.keras.layers.Masking(mask_value = MASKVAL)

    def call(self,x):
        x = self.mask(x)
        return x

class BaseAttention(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()

class CrossAttention(BaseAttention):
    def call(self,x,context):
        attn_output, attn_scores = self.mha(query = x, key = context, 
        value = context,
        return_attention_scores = True)
    
        self.last_attn_scores = attn_scores

        x = self.add([x,attn_output])
        x = self.layernorm(x)

        return x

class GlobalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(
            query = x,
            value = x,
            key = x
        )
        x = self.add([x,attn_output])
        x = self.layernorm(x)
        return x

class CausalSelfAttention(BaseAttention):
    def call(self,x):
        attn_output = self.mha(
            query = x,
            value = x,
            key = x,
            use_causal_mask = True)
        x = self.add([x,attn_output])
        x = self.layernorm(x)
        return x

class FeedForward(tf.keras.layers.Layer):
    def __init__(self,d_model, dff, dropout = 0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(dff,activation = 'elu'),
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
    def __init__(self, *, d_model, num_heads, dff, dropout_rate = 0.1):
        super().__init__()
    
        self.self_attention = GlobalSelfAttention(
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
    def __init__(self, *, num_layers, d_model, num_heads,MASKVAL, dff, dropout_rate = 0.1):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.ParticleMasker = ParticleMasker(MASKVAL)
        
        self.encoder_layers = [
            EncoderLayer(d_model=d_model,
            num_heads = num_heads,
            dff = dff,
            dropout_rate = dropout_rate
            )
        for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(dropout_rate)
    
    def call(self, x):
        x = self.ParticleMasker(x)

        x = self.dropout(x)

        for increment in range(self.num_layers):
            x = self.encoder_layers[increment](x)
        
        return x
    
class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self,
               *,
               d_model,
               num_heads,
               dff,
               dropout_rate=0.1):
    super(DecoderLayer, self).__init__()

    self.causal_self_attention = CausalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.cross_attention = CrossAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.ffn = FeedForward(d_model, dff)

  def call(self, x, context):
    x = self.causal_self_attention(x=x)
    x = self.cross_attention(x=x, context=context)

    self.last_attn_scores = self.cross_attention.last_attn_scores

    x = self.ffn(x)
    return x

class Decoder(tf.keras.layers.Layer):
  def __init__(self, *, num_layers, d_model, num_heads, dff, MASKVAL = MASKVAL,
               dropout_rate=0.1):
    super(Decoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers
    self.ParticleMasker = ParticleMasker(MASKVAL=MASKVAL)

    self.dropout = tf.keras.layers.Dropout(dropout_rate)
    self.dec_layers = [
        DecoderLayer(d_model=d_model, num_heads=num_heads,
                     dff=dff, dropout_rate=dropout_rate)
        for _ in range(num_layers)]

    self.last_attn_scores = None

  def call(self, x, context):
    x = self.ParticleMasker(x)
    x = self.dropout(x)

    for increment in range(self.num_layers):
      x  = self.dec_layers[increment](x, context)

    self.last_attn_scores = self.dec_layers[-1].last_attn_scores

    return Sum()(x)   

class Transformer(tf.keras.Model):
    def __init__(self, *, num_layers, d_model, num_heads, dff, n_targets, MASKVAL = MASKVAL, dropout_rate = 0.1):
        super().__init__()
        
        self.encoder = Encoder(num_layers=num_layers, d_model = d_model, num_heads= num_heads, dff = dff,MASKVAL=MASKVAL)

        self.decoder = Decoder(num_layers= num_layers, d_model = d_model, num_heads= num_heads, MASKVAL = MASKVAL, dff=dff, dropout_rate=dropout_rate)

        self.jet_layer = tf.keras.layers.Dense(n_targets+n_targets*(n_targets+1)//2)

    def call(self, inputs):
        context, x = inputs
        
        context = self.encoder(context)

        x = self.decoder(x, context)

        logits = self.jet_layer(x)
        
        return logits
