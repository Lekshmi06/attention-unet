import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Add, Activation, Multiply
from tensorflow.keras.models import Model

def attention_gate(x, g, inter_channel):
    """
    Attention Gate mechanism for the U-Net.
    
    Args:
        x (tensor): Input from the encoder skip connection.
        g (tensor): Input from the decoder (gating signal).
        inter_channel (int): Number of channels for the intermediate convolution.
        
    Returns:
        The output of the attention gate, which is an attended version of x.
    """
    # Gating signal processing
    theta_g = Conv2D(inter_channel, kernel_size=(1, 1), strides=(1, 1), padding='same')(g)
    
    # Skip connection processing
    phi_x = Conv2D(inter_channel, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    
    # Combine and process
    add_xg = Add()([phi_x, theta_g])
    relu_xg = Activation('relu')(add_xg)
    psi = Conv2D(1, kernel_size=(1, 1), strides=(1, 1), padding='same')(relu_xg)
    sigmoid_psi = Activation('sigmoid')(psi)
    
    # Multiply the attention map with the skip connection
    return Multiply()([x, sigmoid_psi])

def build_attention_unet(input_shape=(256, 256, 1)):
    """
    Builds the Attention U-Net model.
    """
    inputs = Input(input_shape)

    # Encoder
    c1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    # Bottleneck
    c5 = Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    # Decoder
    u6 = UpSampling2D((2, 2))(c5)
    att6 = attention_gate(c4, u6, 512)
    m6 = concatenate([u6, att6], axis=3)
    c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(m6)
    c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = UpSampling2D((2, 2))(c6)
    att7 = attention_gate(c3, u7, 256)
    m7 = concatenate([u7, att7], axis=3)
    c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(m7)
    c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = UpSampling2D((2, 2))(c7)
    att8 = attention_gate(c2, u8, 128)
    m8 = concatenate([u8, att8], axis=3)
    c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(m8)
    c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = UpSampling2D((2, 2))(c8)
    att9 = attention_gate(c1, u9, 64)
    m9 = concatenate([u9, att9], axis=3)
    c9 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(m9)
    c9 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model