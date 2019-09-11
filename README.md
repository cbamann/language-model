## Simple LSTM language model

Simple LSTM language model assuming that words are independent given the recurrent hidden state. A new hidden state is computed given the last hidden state and last word, and the next word is predicted given the hidden state.

Pretrained word embeddings (word2vec) is used to accelerate training as words will come already with some useful representation. Predictive performance is measured in terms of perplexity.