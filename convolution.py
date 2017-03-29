# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division

import argparse
import sys

import tensorflow as tf

FLAGS = None

dictionaries = ['english', 'german', 'latin', 'esperanto']
file_ext = '.txt'

def read_data():
    data = []
    labels = []
    for i in range(len(dictionaries)):
        lines = [line.rstrip('\n') for line in open('./data/' + dictionaries[i] + file_ext)]
        with open('./data/' + dictionaries[i] + file_ext) as fp:
            for line in fp:
                characters = [ord(char) - 96 for char in line.rstrip('\n')]
                for j in range(10 - len(characters)):
                    characters.append(0)
                data.append(characters)
                language = [0 for _ in range(4)]
                language[i] = 1
                labels.append(language)
    return data, labels

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# convolve with a stride of 1 (increment by 1 character), same output size
def conv1d(x, W):
    return tf.nn.conv1d(x, W, stride=1, padding='SAME')

"""
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')
"""

def main(_):
    # Import data
    raw_data, raw_labels = read_data()

    # expected word size of 10 letters max
    x = tf.placeholder(tf.float32, [None, 10])
    y_ = tf.placeholder(tf.float32, [None, 4])

    # first convolution layer
    W_conv1 = weight_variable([3, 1, 32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(x, [-1,10,1])

    h_conv1 = tf.nn.relu(conv1d(x_image, W_conv1) + b_conv1)
    #h_pool1 = max_pool_2x2(h_conv1)

    # second convolution
    W_conv2 = weight_variable([3, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv1d(h_conv1, W_conv2) + b_conv2)
    #h_pool2 = max_pool_2x2(h_conv2)

    # fully connected layer
    W_fc1 = weight_variable([10 * 64, 10])
    b_fc1 = bias_variable([10])

    h_pool2_flat = tf.reshape(h_conv2, [-1, 10*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # readout layer (11 word inputs to 4 languages)
    W_fc2 = weight_variable([10, 4])
    b_fc2 = bias_variable([4])

    # output function
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # globali variable initialization
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # loss function
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # prediction acurracy functions
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # initialize placeholders
    sess.run(tf.global_variables_initializer())

    # input queues
    input_data = tf.constant(raw_data)
    input_labels = tf.constant(raw_labels)
    word, language = tf.train.slice_input_producer(
            [input_data, input_labels],
            shuffle=True)

    # create batches
    batch = tf.train.batch([word, language], 50)

    # start input threads
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    #for i in 100:
    words, languages = sess.run([batch[0], batch[1]])
    train_step.run(feed_dict={x: words, y_: languages, keep_prob: 0.5})
    """
    if i%50 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    """
    """
    train_accuracy = accuracy.eval(feed_dict={
        x:words[0], y_: labels[0], keep_prob: 1.0})
    """

    """
    print("test accuracy %g"%accuracy.eval(feed_dict={
        x: data, y_: labels, keep_prob: 1.0}))
    """

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
