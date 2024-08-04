from __future__ import division, print_function, unicode_literals

import numpy as np
import os

import matplotlib
import matplotlib.pyplot as plt

import tensorflow as tf

print("--- start ---")

n_inputs =3
n_neurons = 5

X0 = tf.placeholder(tf.float32, [None, n_inputs])
X1 = tf.placeholder(tf.float32, [None, n_inputs])

basic_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons)
output_seqs, states = tf.nn.static_rnn(basic_cell, [X0, X1], dtype=tf.float32)

Y0, Y1 = output_seqs

print("--- end ---")

