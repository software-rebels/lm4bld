from nltk.lm import AbsoluteDiscountingInterpolated
from nltk.lm import KneserNeyInterpolated
from nltk.lm import Lidstone

from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm.vocabulary import Vocabulary

import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

class NGramModel:
    def __init__(self, order):
        self.order = order
        self.model = Lidstone(0.00017, self.order)
        empty_vocab = Vocabulary(unk_cutoff=1)

    def fit(self, train_sents):
        train_corp, vocab = padded_everygram_pipeline(self.order, train_sents)
        self.model.fit(train_corp, vocab)

    def getGrams(self, sents):
        ngrams = list()

        for sent in sents:
            paddedTokens = list(pad_both_ends(sent, n=self.order))
            ngrams += list(everygrams(paddedTokens, max_len=self.order))

        return ngrams

    def crossEntropy(self, sents):
        return self.model.entropy(self.getGrams(sents))

    def unkRate(self, sents):
        ngrams = self.getGrams(sents)
        unk_count = 0

        for gram in ngrams:
            gram_or_unk = self.model.vocab.lookup(gram)
            if (gram_or_unk[0] == "<UNK>"):
                #print(gram)
                unk_count += 1

        return unk_count / len(ngrams)
    
    def guessNextToken(self, context, nCandidates):
        sorted_sample = sorted(self.model.context_counts(context))
        return sorted_sample[0:nCandidates]

class LSTMModel:
    def __init__(self, order):
        self.tokenizer = Tokenizer()

    def doFit(self, X, y, max_length):
        model = Sequential()
        model.add(Embedding(self.vocab_size, 10,
                            input_length=max_length-1))
        model.add(LSTM(300))
        model.add(Dense(self.vocab_size, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['categorical_crossentropy'])

        model.fit(X, y, batch_size=20, epochs = 13, verbose = 2)

        return model

    def getSequences(self, sents):
        sequences = list()

        for sent in sents:
            encoded = self.tokenizer.texts_to_sequences(sent)[0]
            for i in range(1, len(encoded)):
                sequences.append(encoded[:i+1])

        # Pad sequences
        max_length = self.length if self.length is not None else max([len(seq) for seq in sequences])
        return pad_sequences(sequences, maxlen=max_length, padding='pre')

    def seqsToArrays(self, sequences):
        sequences = array(sequences)
        X, y = sequences[:,:-1],sequences[:,-1]
        y = to_categorical(y, num_classes=self.vocab_size)

    def fit(self, train_sents):
        # Transform sents
        self.tokenizer.fit_on_texts(train_sents)
        self.vocab_size = len(self.tokenizer.word_index) + 1

        sequences = self.getSequences(train_sents)
        X, y = self.seqsToArrays(sequences)

        self.length = len(sequences[0])

        # Fit LSTM RNN
        self.model = self.doFit()

    def crossEntropy(self, sents):
        sequences = self.getSequences(sents)
        X, y = self.seqsToArrays(sequences)
        return self.model.evaluate(X, y, verbose=1)[0]

    def unkRate(self, sents):
        return -1

    def guessNextToken(self, context, nCandidates):
        encoded = self.tokenizer.texts_to_sequences(context)[0]
        encoded = pad_sequences([encoded], maxlen=self.length, padding='pre')
        yhat = model.predict(encoded, verbose=0)
