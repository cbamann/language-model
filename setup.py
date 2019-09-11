#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os

# DATA packaged
import numpy as np
import pandas as pd
from pandas import Series, DataFrame

# ML packages
import tensorflow as tf

from keras import backend as K
K.clear_session

# Word embedding
import gensim
#from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors

# Text tockenization
from nltk.tokenize import sent_tokenize, word_tokenize

# Miscellaneous
from random import sample
from functools import reduce
from collections import Counter
import itertools # itertools.repeat(x, 3)


###############################################################################

global FOLDER_NN_MODELS, DATA_FOLDER 

# Directory of the folder where data and word embeddings are located
PROJECT_FOLDER = "./"
DATA_FOLDER = PROJECT_FOLDER + "data/"
FOLDER_NN_MODELS = PROJECT_FOLDER + "nn_models/"

global NUM_FOR_TEST # How many batches to use for testing
NUM_FOR_TEST = 64*5 

# READ AND PREPROCESS LOCAL FILES
exec(open(PROJECT_FOLDER + "read_sentences.py").read())


###############################################################################
# Network parameters

flags = tf.app.flags
FLAGS = flags.FLAGS

# General Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of word embedding (default: 300)")
tf.flags.DEFINE_integer("vocab_size", 20000, "Vocabulary")
tf.flags.DEFINE_integer("sent_len", 30, "Maximum sentence length")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("clip_gradient", 5, "Clip the norm of the gradients to 5")
tf.flags.DEFINE_float("learning_rate", 0.001, "Default Adam learning rate")

# RNN hyperparameters
tf.flags.DEFINE_integer("hidden_units", 512, "The size of the hidden cell layer")
tf.flags.DEFINE_integer("hidden_units_large", 1024, "The size of the hidden cell layer")
#tf.flags.DEFINE_float('learning_rate', 0.01, 'Learning rate for the optimization algorithms')

# Session Configuraion parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# TBD
tf.flags.DEFINE_integer("intra_op_parallelism_threads", 4, "Nodes that can use multiple threads to parallelize their execution will schedule the individual pieces into this pool.")
tf.flags.DEFINE_integer("inter_op_parallelism_threads", 4, "All ready nodes are scheduled in this pool.")

tf.flags.DEFINE_integer("intra_op_parallelism_threads_test", 1, "Nodes that can use multiple threads to parallelize their execution will schedule the individual pieces into this pool.")
tf.flags.DEFINE_integer("inter_op_parallelism_threads_test", 1, "All ready nodes are scheduled in this pool.")

session_conf_cluster = tf.ConfigProto(    
      allow_soft_placement = FLAGS.allow_soft_placement,
      log_device_placement = FLAGS.log_device_placement,
      intra_op_parallelism_threads = FLAGS.intra_op_parallelism_threads, 
      inter_op_parallelism_threads = FLAGS.inter_op_parallelism_threads,    
)

session_conf_test = tf.ConfigProto(    
      allow_soft_placement = FLAGS.allow_soft_placement,
      log_device_placement = FLAGS.log_device_placement,
      intra_op_parallelism_threads = FLAGS.intra_op_parallelism_threads_test, 
      inter_op_parallelism_threads = FLAGS.inter_op_parallelism_threads_test,    
)


###############################################################################

def prepare_batch(df_inp, 
                  batch_size = FLAGS.batch_size, 
                  sent_len = FLAGS.sent_len, 
                  null_elem = vocab_dict["<pad>"]):
    """
    prepare standardized batches
    
    Example:
        df_inp = train_df_enc[: 46,:]
        df_out, added = prepare_batch(df_inp)
    
    """
    df_out, added = df_inp, 0
    
    if len(df_inp) < batch_size:
        
        added = batch_size - len(df_inp)
        tmp = null_elem * np.ones((added, FLAGS.sent_len))
        df_out = np.concatenate((df_inp, tmp), axis=0)
        
    return (df_out, added)
        
