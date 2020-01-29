#available
def available():
    import tensorflow as tf    
    return tf.test.is_gpu_available()

#list devices
def devices():
    from tensorflow.python.client import device_lib
    return device_lib.list_local_devices()

available()

devices()