from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras.models import Model

from autoencoder_base import AutoencoderBase
import numpy as np

class ConvolutionalAutoencoder(AutoencoderBase):

    def __init__(self, w_in, h_in, dims_conv, dims_full):
        resize_factor = len(dims_conv)

        enc_dim = dims_full[-1]

        input_img = Input(shape=(w_in, h_in, 1), name='EncoderIn')
        decoder_input = Input(shape=(enc_dim, ), name='DecoderIn')

        flatten_filters = 1
        if len(dims_conv) > 1:
            flatten_filters = dims_conv[-1][0]

        flatten_input = (w_in, h_in, flatten_filters)
        for i in range(resize_factor):
            flatten_input = (flatten_input[0] // 2 + flatten_input[0] % 2, flatten_input[1] // 2 + flatten_input[1] % 2, flatten_input[2])

        flatten_output = (np.prod(flatten_input),)


        encoded = input_img

        #encoder conv layers
        for i, (filters, rows, cols) in enumerate(dims_conv):
            name = 'Conv{0}'.format(i)
            encoded = Convolution2D(filters, rows, cols, activation='relu', border_mode='same', name=name)(encoded)
            encoded = MaxPooling2D((2, 2), border_mode='same', name= 'MaxPool{0}'.format(i))(encoded)

        encoded = Flatten() (encoded)

        #encoder fully connected layers
        for i, dim in enumerate(dims_full):
            name = 'FullEnc{0}'.format(i)
            encoded = Dense(dim, activation='relu') (encoded)

        decoded = encoded
        decoder = decoder_input

        #decoder fully connected layers
        for i, dim in enumerate(reversed(flatten_output + tuple(dims_full[:-1]))):
            name = 'FullDec{0}'.format(i)
            fullLayer = Dense(dim, activation='relu')

            decoded = fullLayer(decoded)
            decoder = fullLayer(decoder)

        reshapeLayer = Reshape(flatten_input, input_shape=flatten_output)
        decoded = reshapeLayer(decoded)
        decoder = reshapeLayer(decoder)

        #decoder conv layers
        for i, (filters, rows, cols) in enumerate(reversed(dims_conv)):
            convlayer = Convolution2D(filters, rows, cols, activation='relu', border_mode='same', name='Deconv{0}'.format(i))
            decoded = convlayer(decoded)
            decoder = convlayer(decoder)

            upsample = UpSampling2D((2, 2), name='UpSampling{0}'.format(i))
            decoded = upsample(decoded)
            decoder = upsample(decoder)

        if len(dims_conv) > 0:
            convlayer = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same')
            decoded = convlayer(decoded)
            decoder = convlayer(decoder)

        self.autoencoder = Model(input=input_img, output=decoded)
        self.encoder = Model(input=input_img, output=encoded)
        self.decoder = Model(input=decoder_input, output=decoder)   

        self.autoencoder.compile(optimizer='adam', loss='mean_squared_error')