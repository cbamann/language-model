#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###############################################################################
# GLOBAL IMPORTS

import numpy as np
import pandas as pd
from pandas import Series, DataFrame

from collections import Counter
import itertools

# Folder where all data is located
global DATA_FOLDER
global MAX_LEN, VOCAB_SIZE

MAX_LEN = 30
VOCAB_SIZE = 20000


###############################################################################
# LOAD AND PREPROCESS

train_df = open(DATA_FOLDER + "sentences.train", "r").readlines()
test_df = open(DATA_FOLDER + "sentences_test.txt", "r").readlines()
eval_df = open(DATA_FOLDER + "sentences.eval", "r").readlines()
predict_df = open(DATA_FOLDER + "sentences.continuation", "r").readlines()

#-----------------------------------------------------------------------------#

train_df = [str(l).replace("\n",  "") for l in train_df ]
test_df = [str(l).replace("\n",  "") for l in test_df ]
eval_df = [str(l).replace("\n",  "") for l in eval_df ]
predict_df = [str(l).replace("\n",  "") for l in predict_df ]

print("Number of {} {} {} {}".format(len(train_df), len(test_df), 
      len(eval_df), len(predict_df)))


###############################################################################
# DEFINE VOCABULARY

train_df_merged = "".join(train_df)
train_df_merged = train_df_merged.replace(".\n", " <eos> <bos> ")
train_df_merged = train_df_merged.replace(".", " <eos> <bos> ")
train_df_merged = train_df_merged.replace("\n", " <eos> <bos> ")

words_train = Counter(train_df_merged.split(" "))

###############################################################################
            
global vocab_dict, vocab_inv

vocab = words_train.most_common(VOCAB_SIZE - 2)

vocab_dict = {w:(i + 2) for (i , w) in enumerate([w for (w, n) in vocab])}
vocab_dict["<pad>"] = 0
vocab_dict["<unk>"] = 1
vocab_inv = list(vocab_dict.keys())


###############################################################################
# Local Helper Functions

def encode_input(inp_df = train_df, 
                 dict_inp = vocab_dict,
                 max_len = MAX_LEN):
    """
    sentence -> array of integers [w_id1, w_id2, w_id3, ... w_id30]
    
    Args: 
        index_list (np.array/list): indexes of the sentences to be converted
        inp_df              (list): list of
        dict_inp            (dict): {"w" (str) : w_id (int)}
        max_len              (int): maximum length of a sentence
            
    Example:
        tst = encode_input(inp_df=train_df[:10])
    """

    x_out = []
    
    for i in range(len(inp_df)):
        
        if i < len(inp_df):
            
            sentence = inp_df[i]
            len_sent = len( sentence.split(" ") )
                
            if len_sent <= max_len - 2:
                
                tmp = [ vocab_dict["<bos>"] ]
                for word in sentence.split(" "):  
                    tmp.append( dict_inp.get( word, 1 )) # integerize 1 corresponds to 
                tmp.append( vocab_dict["<eos>"] ) 

                if len(tmp) < max_len:
                    tmp = tmp + list( itertools.repeat(dict_inp["<pad>"], max_len - len(tmp) ) )
        
                x_out.append(tmp) 

    x_out = np.array(x_out)
    return x_out


###############################################################################

train_df_enc = encode_input(train_df)
test_df_enc = encode_input(test_df)
eval_df_enc = encode_input(eval_df)
predict_df_enc = encode_input(predict_df)

print("Deleted {} {} {} sentences".format( 
      len(train_df) - len(train_df_enc), 
      len(test_df) - len(test_df_enc), 
      len(eval_df) - len(eval_df_enc)))

