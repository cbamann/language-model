#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Postprocessing functions
    1) compute perplexity
    2) predict one step ahead
"""

import tensorflow as tf
PROJECT_FOLDER = "./"

# READ AND perpROCESS LOCAL FILES
exec(open(PROJECT_FOLDER + "setup.py").read())

#-----------------------------------------------------------------------------# 

def compute_perplexity(fname_model, 
                       inp_df = test_df_enc, 
                       dict_inp = vocab_dict,
                       test_mode = True ):
    """
    Compute perplexity of all sentences in test_df    
    
    Args:
        fname_model (str): location of the model          
    """
    
    sess = tf.Session(config=session_conf_test)
    saver = tf.train.Saver()
    
    no_of_batches = int(len(inp_df)/FLAGS.batch_size)
    
    if test_mode:       
        no_of_batches = 50
       
    # restore session         
    with sess.as_default():
            
	 saver.restore(sess, fname_model)        
        
	 # feed different batches to network
        print("No. of batches: {}".format(no_of_batches))
        
        ptr = 0
        perplexity = []
        
        for j in range(no_of_batches-1):

            print("Batch: {}".format(j))
            
            inp_vals = inp_df[ptr: min([ptr + FLAGS.batch_size, len(inp_df)-1 ])]
            len_orig = len(inp_vals)
            inp_vals , added =  perpare_batch(inp_vals)
            l = sess.run(loss, feed_dict = {x_model: inp_vals})

            # divide each loss by sentence len:
            snt_ln = [sum(inp_vals[i] != 0) for i in range(len(inp_vals))]
            perpl = [np.exp(l[i] / snt_ln[i]) for i in range(len(inp_vals))]
            
            if added > 0: # exclude perplexity corresponding to added sells                
                print("Added  {} lines".format(added))
                perpl = perpl[:len_orig]
            
            print("Average perplexity: {}".format(np.mean(perpl)))
            perplexity = perplexity + list(perpl)
            
            ptr += FLAGS.batch_size
            
    return perplexity

#-----------------------------------------------------------------------------# 
# script  to create output files
    
prp = compute_perplexity(FOLDER_NN_MODELS+"exp", test_mode = False  )

