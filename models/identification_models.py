import tensorflow as tf
from tensorflow import keras
from keras import layers, models,activations, initializers , Model
import numpy as np


## ------------------ HELPER FUNCTIONS & CLASSES -----------------------##
##                                                                      ##
##                                                                      ## 
## ---------------------------------------------------------------------##
def conv_block(filters,kernel_size,initializer,poolsize,input):
    x = layers.Conv1D(filters = filters, 
                               kernel_size=kernel_size,
                               kernel_initializer=initializer,
                               )(input)
    x = layers.MaxPooling1D(pool_size=poolsize)(x)
    x = layers.BatchNormalization()(x)
    return layers.Activation('relu')(x)

def convBN(filters,kernel_size,initializer,input):
    x = layers.Conv1D(filters = filters, 
                               kernel_size=kernel_size, 
                               padding='same',
                               kernel_initializer=initializer,
                               )(input) 
    x = layers.BatchNormalization()(x)
    return layers.Activation('relu')(x)

def denseBN(units,initializer,input):
    x = layers.Dense(units=units,
                               kernel_initializer=initializer,
                               )(input) 
    x = layers.BatchNormalization()(x)
    return layers.Activation('relu')(x)

def conv_resblock(filters,kernel_size,initializer,input,match_dims):
    x_skip = input
    if match_dims:
        x = layers.Conv1D(filters = filters, 
                                kernel_size=kernel_size, 
                                padding='same',
                                strides=2,
                                kernel_initializer=initializer,
                                )(input) 
    else:
        x = layers.Conv1D(filters = filters, 
                                kernel_size=kernel_size, 
                                padding='same',
                                kernel_initializer=initializer,
                                )(input) 
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv1D(filters = filters, 
                               kernel_size=kernel_size, 
                               padding='same',
                               kernel_initializer=initializer,
                               )(x) 
    x = layers.BatchNormalization()(x)

    if match_dims:
        x_skip = layers.Lambda(lambda x: tf.pad(x[:,::2,:], tf.constant([[0,0],[0,0],[filters//4,filters//4]]),mode='CONSTANT'))(input)
  
    x = layers.Add()([x_skip,x])
    return tf.keras.layers.Activation('relu')(x)

##------------ transformers helper function --------------- ##
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


class StochasticDepth(layers.Layer):
    def __init__(self, drop_prop, **kwargs):
        super(StochasticDepth, self).__init__(**kwargs)
        self.drop_prob = drop_prop

    def call(self, x, training=None):
        if training:
            keep_prob = 1 - self.drop_prob
            shape = (tf.shape(x)[0],) + (1,) * (tf.shape(x).shape[0] - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x
    
class CCTTokenizer(layers.Layer):
    def __init__(
        self,
        num_output_channels,
        kernel_size=32,
        stride=1,
        pooling_kernel_size=2,
        pooling_stride=None,
        positional_emb=True,
        **kwargs,
    ):
        super(CCTTokenizer, self).__init__(**kwargs)

        # This is our tokenizer.
        self.conv_model = keras.Sequential()
        for num_out_chan in num_output_channels:
            self.conv_model.add(
                layers.Conv1D(
                    num_out_chan,
                    kernel_size,
                    stride,
                    padding="valid",
                    use_bias=False,
                    activation="relu",
                    kernel_initializer="he_normal",
                )
            )
            self.conv_model.add(
                layers.MaxPool1D(pooling_kernel_size, pooling_stride)
            )
        
        self.positional_emb = positional_emb

    def call(self, images):
        outputs = self.conv_model(images)
        # After passing the images through our mini-network the spatial dimensions
        # are flattened to form sequences.
        reshaped = tf.reshape(
            outputs,
            (-1,tf.shape(outputs)[1],tf.shape(outputs)[-1]),
        )
        return reshaped

    def positional_embedding(self, spec_length):
        # Positional embeddings are optional in CCT. Here, we calculate
        # the number of sequences and initialize an `Embedding` layer to
        # compute the positional embeddings later. 
        if self.positional_emb:
            dummy_inputs = tf.ones((1,spec_length,1))
            dummy_outputs = self.call(dummy_inputs)
            sequence_length = tf.shape(dummy_outputs)[1]
            projection_dim = tf.shape(dummy_outputs)[-1]

            embed_layer = layers.Embedding(
                input_dim=sequence_length, output_dim=projection_dim
            )
            return embed_layer, sequence_length
        else:
            return None

class Patches(layers.Layer):
    def __init__(self, patch_size,patch_stride):
        super(Patches, self).__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride

    def call(self, spec):
        batch_size = tf.shape(spec)[0]
        spec = tf.expand_dims(spec, axis = 1)
        spec = tf.expand_dims(spec, axis = 3)
        patches = tf.image.extract_patches(
                                    images=spec,
                                    sizes=[1,1,self.patch_size,1],
                                    strides=[1,1,self.patch_stride,1],
                                    rates=[1,1,1,1],
                                    padding="VALID",
                                        )
        patches = tf.reshape(patches, [batch_size, -1,self.patch_size])
        return patches

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

## --------------------------- MODELS ----------------------------------##
##                                                                      ##
##                                                                      ## 
## ---------------------------------------------------------------------##

def CNN(spectrum_length,N_elem):
    '''
    CNN-MLP model inspired to be a 1D version of VGG-11
    '''
    
    initializer = 'he_normal'
    activ = activations.relu

    input = layers.Input(shape = (spectrum_length,1))
    x = layers.Conv1D(filters=64,kernel_size= 32, activation=activ, padding='same',kernel_initializer=initializer)(input)
    x = layers.MaxPool1D(pool_size=2)(x)
    x = layers.Conv1D(filters=128,kernel_size=32, activation=activ, padding='same',kernel_initializer=initializer)(x)
    x = layers.MaxPool1D(pool_size=2)(x)
    x = layers.Conv1D(filters=256,kernel_size=32, activation=activ, padding='same',kernel_initializer=initializer)(x)
    x = layers.Conv1D(filters=256,kernel_size = 32, activation=activ, padding='same',kernel_initializer=initializer)(x)
    x = layers.MaxPool1D(pool_size=2)(x)
    x = layers.Conv1D(filters=512,kernel_size=32, activation=activ, padding='same',kernel_initializer=initializer)(x)
    x = layers.Conv1D(filters=512,kernel_size = 32, activation=activ, padding='same',kernel_initializer=initializer)(x)
    x = layers.MaxPool1D(pool_size=2)(x)
    x = layers.Conv1D(filters=512,kernel_size=32, activation=activ, padding='same',kernel_initializer=initializer)(x)
    x = layers.Conv1D(filters=512,kernel_size = 32, activation=activ, padding='same',kernel_initializer=initializer)(x)
    x = layers.MaxPool1D(pool_size=2)(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(rate=0.5)(x)
    x = layers.Dense(256, activation=activ, kernel_initializer=initializer)(x)
    x = layers.Dropout(rate=0.5)(x)
    x = layers.Dense(256, activation=activ, kernel_initializer=initializer)(x)
    output = layers.Dense(N_elem, activation =activations.sigmoid,kernel_initializer=initializer)(x)

    model = Model(inputs = input, outputs = output)
    return model




def ResNet(spectrum_length,N_elem,reduction_method):
    KS = 32
    initializer = "he_normal"

    input = layers.Input(shape = (spectrum_length,1))
    x = layers.Conv1D(filters = 16, kernel_size=KS, padding='same',kernel_initializer=initializer)(input) 
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)


    x = conv_resblock(filters = 16, kernel_size=KS,initializer=initializer, input=x,match_dims=False)
    x = conv_resblock(filters = 16, kernel_size=KS,initializer=initializer, input=x,match_dims=False)                   
    x = conv_resblock(filters = 32, kernel_size=KS,initializer=initializer, input=x,match_dims=True)
    x = conv_resblock(filters = 32, kernel_size=KS,initializer=initializer, input=x,match_dims=False)
    x = conv_resblock(filters = 32, kernel_size=KS,initializer=initializer, input=x,match_dims=False)
    x = conv_resblock(filters = 64, kernel_size=KS,initializer=initializer, input=x,match_dims=True)
    x = conv_resblock(filters = 64, kernel_size=KS,initializer=initializer, input=x,match_dims=False)
    x = conv_resblock(filters = 64, kernel_size=KS,initializer=initializer, input=x,match_dims=False)
    x = conv_resblock(filters = 64, kernel_size=KS,initializer=initializer, input=x,match_dims=False)
    x = conv_resblock(filters = 128, kernel_size=KS,initializer=initializer, input=x,match_dims=True)
    x = conv_resblock(filters = 128, kernel_size=KS,initializer=initializer, input=x,match_dims=False)
    x = conv_resblock(filters = 128, kernel_size=KS,initializer=initializer, input=x,match_dims=False)
    x = conv_resblock(filters = 256, kernel_size=KS,initializer=initializer, input=x,match_dims=True)
    x = conv_resblock(filters = 256, kernel_size=KS,initializer=initializer, input=x,match_dims=False)
    x = conv_resblock(filters = 512, kernel_size=KS,initializer=initializer, input=x,match_dims=True)
    x = conv_resblock(filters = 512, kernel_size=KS,initializer=initializer, input=x,match_dims=False)

    if reduction_method == "1x1conv":
        x = layers.Conv1D(filters=1,kernel_size=1,activation='relu',kernel_initializer=initializer)(x)
        x = layers.Reshape((x.get_shape()[1],))(x)
    elif reduction_method == "flatten":
        x = layers.Flatten()(x)
    elif reduction_method == "GAP":
        x = layers.GlobalAveragePooling1D()(x)
    else:
        print("invalid reduction method, must be 1x1conv or flatten or GAP")
    output = layers.Dense(N_elem, activation =activations.sigmoid,kernel_initializer=initializer)(x)
        
    model = Model(inputs = input, outputs = output)
    return model 




def UNet(spectrum_length,N_elem,reduction_method):
    
    initializer = "he_normal"

    input = layers.Input(shape = (spectrum_length,1)) #3072 x 1
    x1_1 = convBN(filters = 16, kernel_size=32,initializer=initializer, input=input)   #3072 x 16           

    x2_1 = layers.MaxPool1D(pool_size = 2)(x1_1) #1536 x 16
    x2_1 = convBN(filters = 32, kernel_size=32,initializer=initializer, input=x2_1) #1536 x 32

    x3_1 = layers.MaxPool1D(pool_size = 2)(x2_1) #768 x 32
    x3_1 = convBN(filters = 64, kernel_size=32,initializer=initializer, input=x3_1) #768 x 64


    x4_1 = layers.MaxPool1D(pool_size = 2)(x3_1) #384 x 64
    x4_1 = convBN(filters = 128, kernel_size=32,initializer=initializer, input=x4_1) #384 x 128

    x5 = layers.MaxPool1D(pool_size = 2)(x4_1) #192 x 128
    x5 = convBN(filters = 256, kernel_size=32,initializer=initializer, input=x5) #192 x 256
    x5 = layers.UpSampling1D(2)(x5) #384 x 256
    x5 = convBN(filters = 128, kernel_size=32,initializer=initializer, input=x5) #384 x 128


   
    x4_2 = layers.Concatenate()([x5 , x4_1])  #384 x 
    x4_2 = convBN(filters = 128, kernel_size=32,initializer=initializer, input=x4_2)  #384 x 128
    x4_2 = layers.UpSampling1D(2)(x4_2)  #768 x 128
    x4_2 = convBN(filters = 64, kernel_size=32,initializer=initializer, input=x4_2)  #768 x 64
    
    x3_2 = layers.Concatenate()([x4_2 , x3_1]) #768 x 128
    x3_2 = convBN(filters = 64, kernel_size=32,initializer=initializer, input=x3_2) #768 x 64
    x3_2 = layers.UpSampling1D(2)(x3_2) #1536 x 64
    x3_2 = convBN(filters = 32, kernel_size=32,initializer=initializer, input=x3_2) #1536 x 32

    x2_2 = layers.Concatenate()([x3_2 , x2_1]) #1536 x 64
    x2_2 = convBN(filters = 32, kernel_size=32,initializer=initializer, input=x2_2) #1536x 32
    x2_2 = layers.UpSampling1D(2)(x2_2) #3072 x 32
    x2_2 = convBN(filters = 16, kernel_size=32,initializer=initializer, input=x2_2) #3072 x 16

    x1_2 = layers.Concatenate()([x1_1 , x2_2]) #3072 x 32
    if reduction_method == "1x1conv":
        x = layers.Conv1D(filters=1,kernel_size=1,kernel_initializer=initializer,activation='relu')(x1_2) 
        x = layers.Reshape(target_shape = (spectrum_length,))(x)
    elif reduction_method == "flatten":
        x = layers.Flatten()(x1_2) 
    elif reduction_method == "GAP":
        x = layers.GlobalAveragePooling1D()(x1_2)
    else:
        print("invalid reduction method, must be 1x1conv or flatten or GAP")

    x = denseBN(units = spectrum_length,initializer = initializer,input=x)
    x = layers.Dropout(rate=0.3)(x)
    x = denseBN(units = spectrum_length/2,initializer = initializer,input=x)
    x = layers.Dropout(rate=0.3)(x)
    x = denseBN(units = spectrum_length/4,initializer = initializer,input=x)
    x = layers.Dropout(rate=0.3)(x)
    x = denseBN(units = spectrum_length/8,initializer = initializer,input=x)
    output = layers.Dense(N_elem, activation =activations.sigmoid,kernel_initializer=initializer)(x)
        
    model = Model(inputs = input, outputs = output)
    return model 


def MLP(spectrum_length,N_elem):
    initializer = "he_normal"
    activ = activations.relu

    input = layers.Input(shape = (spectrum_length,))
    x = denseBN(spectrum_length*2,initializer,input)
    x = layers.Dropout(rate = 0.3)(x)
    x = denseBN(spectrum_length,initializer,x)
    x = layers.Dropout(rate = 0.3)(x)
    x = denseBN(spectrum_length,initializer,x)
    x = layers.Dropout(rate = 0.3)(x)
    x = denseBN(spectrum_length/2,initializer,x)
    x = layers.Dropout(rate = 0.3)(x)
    x = denseBN(spectrum_length/2,initializer,x)
    x = layers.Dropout(rate = 0.3)(x)
    x = denseBN(spectrum_length/4,initializer,x)
    x = layers.Dropout(rate = 0.3)(x)
    x = denseBN(spectrum_length/8,initializer,x)
    output = layers.Dense(N_elem, activation=activations.sigmoid,kernel_initializer=initializer)(x)
    
    model = Model(inputs = input, outputs = output)
    return model



def ViT(spectrum_length,N_elem,reduction_method):
    patch_size = 16
    patch_stride = int(patch_size/2)
    projection_dim = 128
    num_patches = int((spectrum_length-patch_size)/patch_stride +1)
    transformer_layers = 8
    mlp_head_units = [512 , 256, 128] 
    num_heads = 16
    transformer_dense_units = [projection_dim * 2,projection_dim]  # Size of the transformer layers
    inputs = layers.Input(shape=[spectrum_length])

    # Create patches.
    patches = Patches(patch_size,patch_stride)(inputs)

    if reduction_method == "token":
        class_token = layers.Dense(units=patch_size)(tf.zeros(shape=(tf.shape(patches)[0],1,tf.shape(patches)[2])))
        patches = layers.concatenate([class_token,patches], axis = 1)
        num_patches = num_patches +1

        
    # Encode patches
    patch_encod = PatchEncoder(num_patches, projection_dim)(patches)


    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(patch_encod)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=patch_size, dropout=0.1)(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, patch_encod])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_dense_units, dropout_rate=0.1)
        # Skip connection 2.
        patch_encod = layers.Add()([x3, x2])


    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(patch_encod)
    if reduction_method == "1x1conv":
        representation = layers.Conv1D(filters=1,kernel_size=1)(representation)
        representation = layers.Reshape((representation.get_shape()[1],))(representation)
    elif reduction_method == "flatten":
        representation = layers.Flatten()(representation)
    elif reduction_method == "GAP":
        representation = layers.GlobalAveragePooling1D()(representation)
    elif reduction_method == "token":
        representation = representation[:,0,:]
    else:
        print(("invalid reduction method, must be 1x1conv or flatten or GAP or token"))
    representation = layers.Dropout(0.1)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units[0:2], dropout_rate=0.1)
    features = layers.Dense(mlp_head_units[2],activation=tf.nn.gelu)(features)
    # Classify outputs.
    out = layers.Dense(N_elem, activation = 'sigmoid')(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=out)
    return model



def CCT(spectrum_length,N_elem,reduction_method):
   
    projection_dim = 128
    num_heads = 8
    transformer_units = [projection_dim*2,projection_dim]
    transformer_layers = 8
    positional_emb = True

    inputs = layers.Input((spectrum_length,1))

    # Encode patches.
    cct_tokenizer = CCTTokenizer(num_output_channels=[projection_dim/4, projection_dim/2, projection_dim,],positional_emb=positional_emb)
    encoded_patches = cct_tokenizer(inputs)

    # Apply positional embedding.
    if positional_emb:
        pos_embed, seq_length = cct_tokenizer.positional_embedding(spectrum_length)
        positions = tf.range(start=0, limit=seq_length, delta=1)
        position_embeddings = pos_embed(positions)
        encoded_patches += position_embeddings

    # Create multiple layers of the Transformer block.
    for i in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)

        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])

        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)

        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)

        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Apply sequence pooling.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    if reduction_method == "1x1conv":
        representation = layers.Conv1D(filters=1,kernel_size=1)(representation)
        representation = layers.Reshape((representation.get_shape()[1],))(representation)
    elif reduction_method == 'seq_pool':
        attention_weights = tf.nn.softmax(layers.Dense(1)(representation), axis=1)
        weighted_representation = tf.matmul(
            attention_weights, representation, transpose_a=True
        )
        representation = tf.squeeze(weighted_representation, -2)
    else:
        print(("invalid reduction method"))

    # Classify outputs.
    output = layers.Dense(N_elem, activation='sigmoid')(representation)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=output)
    return model

