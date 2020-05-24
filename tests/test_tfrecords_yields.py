#Test tfrecords to make sure they load
from utils import tfrecords
import tensorflow as tf
import glob

list_of_tfrecords = glob.glob("/orange/ewhite/b.weinstein/NEON/crops/*.tfrecord")
outputs = tfrecords.create_tensors(list_of_tfrecords, batch_size=1)

sess = tf.Session()

counter = 0
while True:
    try:
        print(counter)
        tf_inputs = sess.run([outputs])
        assert tf_inputs[0].shape == (1,800,800,3)
        counter+=1
    except:
        break

print(counter)
        
    