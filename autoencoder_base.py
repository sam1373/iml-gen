from keras import backend

from keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau

class AutoencoderBase(object):

    def train(self, x_train, x_test, epochs, batch_size, log_dir='/tmp/autoencoder', stop_early=True):
        callbacks = []

        if backend._BACKEND == 'tensorflow':
            callbacks.append(TensorBoard(log_dir=log_dir))

        if stop_early:
            callbacks.append(EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto'))

        #callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1, mode="min", min_lr=1e-4))

        self.autoencoder.fit(x_train, x_train,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=callbacks)

    def encode(self, x):
        return self.encoder.predict(x)

    def decode(self, x):
        return self.decoder.predict(x)
        
    def summary(self):
        self.autoencoder.summary()

