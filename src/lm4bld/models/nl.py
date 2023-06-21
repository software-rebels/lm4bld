from nltk.lm import Lidstone

from nltk.lm.preprocessing import pad_both_ends
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm.vocabulary import Vocabulary
from nltk.util import everygrams

from lm4bld.models.api import Model

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from numpy import array
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

#from tensorflow.keras.losses import BinaryCrossentropy

class NGramModel(Model):
    def __init__(self, order, tokenizer, prefix, tokenprefix, ignore_syntax):
        super().__init__(order, tokenizer, prefix, tokenprefix, ignore_syntax)
        self.model = Lidstone(0.00017, self.order)
        empty_vocab = Vocabulary(unk_cutoff=1)

    def load_sents(self, flist):
        sents = list()

        for f in flist:
            t = self.tokenizer(f, self.prefix, self.tokenprefix,
                               self.ignore_syntax)
            sents += t.load_tokens()

        return sents

    def grammify(self, test_sents):
        ngrams = list()

        for sent in test_sents:
            paddedTokens = list(pad_both_ends(sent, n=self.order))
            ngrams += list(everygrams(paddedTokens, max_len=self.order))

        return ngrams

    def fit(self, trainCorpus, filelevel=True):
        train_sents = self.load_sents(trainCorpus) if filelevel else trainCorpus
        train_corp, vocab = padded_everygram_pipeline(self.order, train_sents)
        self.model.fit(train_corp, vocab)

    def crossEntropy(self, flist):
        ngrams = self.grammify(self.load_sents(flist))
        return self.model.entropy(ngrams)

    def unkRate(self, flist):
        ngrams = self.grammify(self.load_sents(flist))
        unk_count = 0

        for gram in ngrams:
            gram_or_unk = self.model.vocab.lookup(gram)
            if (gram_or_unk[0] == "<UNK>"):
                #print(gram)
                unk_count += 1

        return unk_count / len(ngrams)

    def vocabSize(self):
        return len(self.model.vocab)

    def guessNextTokens(self, testCorpus, nCandidates, filelevel=True):
        correct = {}
        incorrect = {}

        sents = self.load_sents(testCorpus) if filelevel else testCorpus

        for sent in sents:
            context = ("<s>",) * (self.order-1) # Context is initialized with padding up the order
            for token in sent:
                guesses = self.guessNextToken(context, nCandidates)
                if (token in guesses):
                    if len(token) not in correct:
                        correct[len(token)] = list()

                    correct[len(token)].append(token)

                else:
                    if len(token) not in incorrect:
                        incorrect[len(token)] = list()

                    incorrect[len(token)].append(token)

                context = context[1:] + (token,)

        rtn = {}
        allkeys = set(list(correct.keys()) + list(incorrect.keys()))
        for token_len in allkeys:
            n_correct = len(correct[token_len]) if token_len in correct else 0
            n_incorrect = len(incorrect[token_len]) if token_len in incorrect else 0
            rtn[token_len] = [n_correct, n_incorrect]

        return rtn

    def guessNextToken(self, context, nCandidates):
        sorted_sample = sorted(self.model.context_counts(context))
        return sorted_sample[0:nCandidates]

    def globalGuessNextTokens(self, testCorpus, budget):
        budget_remaining = budget
        correct = list()
        incorrect = list()
        max_guesses = 10

        for f in testCorpus:
            sents = self.load_sents([f])
            #sents.pop(0) # Get rid of first line, since it is boilerplate

            for sent in sents:
                context = ("<s>",) * (self.order-1) # Context is initialized with padding up the order

                for token in sent:
                    guesses = self.guessNextToken(context, max_guesses)

                    spent_budget = max_guesses

                    if token in guesses:
                        spent_budget = guesses.index(token)
                        correct.append(token)
                    else:
                        incorrect.append(token)

                    budget_remaining -= spent_budget
                    context = context[1:] + (token,)

                    if budget_remaining <= 0:
                        break

                if budget_remaining <= 0:
                    break

            if budget_remaining <= 0:
                break

        return len(correct), len(incorrect)

class LSTMModel(Model):
    def __init__(self, order, tokenizer, prefix, tokenprefix, ignore_syntax):
        super().__init__(order, tokenizer, prefix, tokenprefix, ignore_syntax)
        self.tflowtokenizer = Tokenizer(filters="", oov_token="<OOV>")
        self.length = None
        self.do_fit = True

    def load_sents(self, flist):
        sents = list()

        for f in flist:
            t = self.tokenizer(f, self.prefix, self.tokenprefix,
                               self.ignore_syntax)
            sents += t.load_tokens()

        return sents

    def create_model(self, X, y):
        model = Sequential()
        model.add(Embedding(self.vocab_size, 10,
                            input_length=self.length-1))
        model.add(LSTM(300))
        model.add(Dense(self.vocab_size, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['categorical_crossentropy'])

        return model

    def getSequences(self, sents):
        sequences = list()

        for sent in sents:
            encoded = self.tflowtokenizer.texts_to_sequences(sent)
            for i in range(1, len(encoded)):
                sequences.append(encoded[:i+1])

        # Pad sequences
        max_length = max([len(seq) for seq in sequences])
        if self.length is not None:
            max_length = max([max_length, self.length])

        return pad_sequences(sequences, maxlen=max_length, padding='pre')

    def seqsToArrays(self, sequences):
        sequences = array(sequences)
        X, y = sequences[:,:-1],sequences[:,-1]
        y = to_categorical(y, num_classes=self.vocab_size)

        return X, y

    def fit(self, train_files):
        train_sents = self.load_sents(train_files)

        # Transform sents
        self.tflowtokenizer.fit_on_texts(train_sents)
        self.vocab_size = len(self.tflowtokenizer.word_index) + 1

        sequences = self.getSequences(train_sents)
        X, y = self.seqsToArrays(sequences)

        self.length = len(sequences[0])

        # Create LSTM RNN
        self.model = self.create_model(X, y)

        # Configure checkpointing
        checkpoint_path = "training/cp.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)

        checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path,
                                              save_weights_only=True, verbose=2)

        if self.do_fit:

            # Fit RNN
            self.model.fit(X, y, batch_size=20, epochs = 13,
                           validation_split=0.25,
                           callbacks=[checkpoint_callback],
                           verbose = 2)

        else:
            self.model.load_weights(checkpoint_path)

    def crossEntropy(self, test_files):
        test_sents = self.load_sents(test_files)

        sequences = self.getSequences(test_sents)
        X, y = self.seqsToArrays(sequences)

        return self.model.evaluate(X, y, verbose=2)[0]

    def unkRate(self, sents):
        return -1

    def vocabSize(self):
        return self.vocab_size

    def guessNextToken(self, context, nCandidates):
        encoded = self.tflowtokenizer.texts_to_sequences(context)[0]
        encoded = pad_sequences([encoded], maxlen=self.length, padding='pre')
        yhat = model.predict(encoded, verbose=0)
