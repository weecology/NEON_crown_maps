import numpy as np
import dask
from distributed import wait
import start_cluster

#new line    
client = start_cluster.GPU_cluster(gpus=2)

#available
def available():
    import tensorflow as tf    
    return tf.test.is_gpu_available()
    
#list devices
def devices():
    from tensorflow.python.client import device_lib
    return device_lib.list_local_devices()

#submit 
future = client.submit(devices)
print(future.result())
    
#submit 
future = client.submit(available)
print(future.result())

def run():
    
    def train_model():
        batch_size = 128
        num_classes = 10
        epochs = 1
        
        # input image dimensions
        img_rows, img_cols = 28, 28
        
        # the data, split between train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)
        
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        
        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
        
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
        
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(x_test, y_test))
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        
        return model
    
    def load_data():
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
        return x_test
    
    import keras
    from keras.datasets import mnist
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Flatten
    from keras.layers import Conv2D, MaxPooling2D
    from keras import backend as K
    
    model = train_model()
    x_test = load_data()
    
    #Compute prediction in batch loop of size 100 (slightly contrived example)
    batch_array = np.split(x_test,100)
    
    results = []
    for batch in batch_array:
        prediction = model.predict_on_batch(batch)
        results.append(prediction)
    
    return results

#results = [ ]
#for x in np.arange(4):
    #result = client.submit(run)
    #results.append(result)

#wait(results)