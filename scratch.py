import tensorflow as tf
from utils import tfrecords
import os
os.chdir("tests/")
#output = tfrecords.create_dataset("/Users/ben/Downloads/2018_JERC_4_748000_3454000_image.tfrecord", batch_size=1)    
#output = tfrecords.create_dataset("tests/output/OSBS_027.tfrecord", batch_size=1)

with tf.Session():
    output = tfrecords.create_tensors("/orange/ewhite/b.weinstein/NEON/crops/2018_JERC_4_748000_3454000_image.tfrecord", batch_size=1)
    
    #Check image shape
    with tf.Session() as sess:
        image = sess.run(output)
        print(image.shape)
  
    output = tfrecords.create_tensors("/orange/ewhite/b.weinstein/NEON/crops/2018_JERC_4_748000_3454000_image.tfrecord", batch_size=1)
    counter = 0
    with tf.Session() as sess: 
        while True:
            try:
                a=sess.run(output)
                print(a.shape)
                counter+=1
            except tf.errors.OutOfRangeError:
                
                break         
print("{} windows".format(counter))