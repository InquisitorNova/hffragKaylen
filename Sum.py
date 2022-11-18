from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

class Sum(Layer):
    """
    Simple Sum Layer

    The job of this Sum is ensure the masking to work properly. Hopefully
    we are dealing with time distributed dense layers that should compute
    the mask on their own.
    The code below overrides the masking done by dense layers.
    """

    #Initialise a constructor that acts as a superclass
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True #Turn on masking
    
    def build(self, input_shape):
        pass
    
    def call(self, x, mask = None):
        if mask is not None:
            x = x * K.cast(mask, K.dtype(x))[:,:,None]
        return K.sum(x,axis = 1)
    
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]

    def compute_mask(self, inputs, mask):
        return None
    
    def get_config(self):
        return {}