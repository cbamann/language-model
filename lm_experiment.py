#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
PROJECT_FOLDER = "./"

# READ AND PREPROCESS LOCAL FILES
exec(open(PROJECT_FOLDER + "setup.py").read())
fname_nn_out = FOLDER_NN_MODELS + "exp"

BATCHES_TO_PROCESS = 500


###############################################################################
# Download word embedding

path_embedding = PROJECT_FOLDER + "wordembeddings-dim100.word2vec"
model_embed = KeyedVectors.load_word2vec_format( path_embedding )

# EMBEDDING MATRIX:
vocab_size = len(vocab_inv)
embeddings = np.zeros( ( FLAGS.vocab_size, FLAGS.embedding_dim ) ) 
# + 2 since added symbols <unk> and <pad>

matches = 0
for k, v in vocab_dict.items():
    if k in model_embed.vocab:
        embeddings[v] = model_embed[k]
        matches += 1
    else:
        embeddings[v] = np.random.uniform(low=-0.25, high=0.25, size=FLAGS.embedding_dim )
        
print("%d words out of %d could be loaded" % (matches, vocab_size))


###############################################################################
#-----------------------------------------------------------------------------#
# DEFINE THE GRAPH

tf.reset_default_graph() # clean up just in case

x_model = tf.placeholder(tf.int64 , shape = (None, FLAGS.sent_len))
# Define weights
W = tf.get_variable(name = "W_out", 
                    shape = (FLAGS.hidden_units, FLAGS.vocab_size),  
                    dtype = tf.float64,
                    initializer = tf.contrib.layers.xavier_initializer())

embedding_matrix = tf.constant(embeddings, 
                               shape = (FLAGS.vocab_size, FLAGS.embedding_dim),
                               dtype = tf.float64)

embedded_x = tf.nn.embedding_lookup(embedding_matrix, x_model)

# prepare input and output sequences
X_series = tf.unstack(embedded_x[:,:-1,:], axis = 1)
Y_series = tf.unstack(x_model[:,1:], axis = 1) 

#-----------------------------------------------------------------------------#
# CONSTRUCT THE NETWORK

lstm = tf.nn.rnn_cell.LSTMCell(num_units = FLAGS.hidden_units,
                               dtype= tf.float64 ) # state_is_tuple=True)

_state = lstm.zero_state(dtype = tf.float64, batch_size = FLAGS.batch_size ) 
        
# unrolling
outputs = []
for i in range(len(X_series)):
    (cell_output, _state) = lstm(X_series[i], _state)
    outputs.append(cell_output)
 
#-----------------------------------------------------------------------------#    
# COMPUTE THE LOSS 

loss = 0.0

for i in range(len(X_series) ): # time dimension
    labels_i = Y_series[i]
    out = outputs[i]
    # This op expects unscaled logits, since it performs a softmax on logits internally for efficiency. 
    # Do not call this op with the output of softmax, as it will produce incorrect results.
    prob_unnorm = tf.matmul(out, W) 
    loss_temp = tf.nn.sparse_softmax_cross_entropy_with_logits( 
                        labels = labels_i, 
                        logits = prob_unnorm)
    
    is_padded = tf.dtypes.cast(tf.not_equal(labels_i, vocab_dict["<pad>"]),
                               dtype=tf.float64 )
    loss_temp = tf.math.multiply(loss_temp, is_padded)
   
    loss += loss_temp # is_padded*loss_temp

#-----------------------------------------------------------------------------# 
# OPTIMIZER

params = tf.trainable_variables()

# tf.reset_default_graph() # if something is wrong
optimizer = tf.train.AdamOptimizer(learning_rate = FLAGS.learning_rate) # default rate
gradients, variables = zip(*optimizer.compute_gradients(loss))
gradients, _ = tf.clip_by_global_norm(gradients, FLAGS.clip_gradient)
train_op = optimizer.apply_gradients(zip(gradients, variables))


###############################################################################
# Train the network

# Input parameters
test_mode = True
train_df = train_df_enc
fname_nn = fname_nn_out

#-----------------------------------------------------------------------------# 

init = tf.global_variables_initializer()
saver = tf.train.Saver() 

batches_total = int(len(train_df_enc)/ FLAGS.batch_size)
test_mode = True

if test_mode:
    sess = tf.Session(config=session_conf_cluster)
    #train_df_loc = train_df[:NUM_FOR_TEST]
    no_of_batches = min([BATCHES_TO_PROCESS, batches_total])
else:
    sess = tf.Session(config=session_conf_cluster)
    no_of_batches = batches_total

# run session
with sess.as_default():
    
    sess.run(init)

    # feed batches to network
    print("No of batches: {}".format(no_of_batches))    
    
    learning_errors = []
    ptr = 0
    
    for j in range(no_of_batches):
        print("Batch: {}".format(j))
        
        x_train = train_df_enc[ptr: min([ptr + FLAGS.batch_size, len(train_df_enc)-1 ])]
        x_train , added =  prepare_batch(x_train)
        _ ,  l  = sess.run([train_op, loss], feed_dict = {x_model: x_train })
        
        ptr += FLAGS.batch_size

    # save the session parameters 
    saver.save(sess, fname_nn)
    

