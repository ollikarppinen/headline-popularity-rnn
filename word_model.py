# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import sys
import os
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional, Activation
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils.data_utils import get_file
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import RMSprop
from sklearn.metrics import confusion_matrix
from datetime import datetime
from math import floor
import util
import json

################### HYPERPARAMS ##########################

settings = {
    'ratio': 0.2,
    'title_max_len': 16,
    'batch_size': 64,
    'epochs': 10,
    'word_min_count': 20,
    'lstm': 16,
    'dropout': 0.5,
    'description': 'lstm 64 dropout 0.5'
}
# Creating model directory
ts = datetime.now()
date_string = '-'.join(list(map(str, [ts.year, ts.month, ts.day, ts.hour, ts.minute])))
file_name = os.path.basename(sys.argv[0]).split('.')[0]
cwd = os.getcwd()
model_dir = os.path.join(cwd, 'checkpoints', date_string)
tensorboard_dir = os.path.join(cwd, 'tensorboard')
try:
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    if not os.path.isdir(tensorboard_dir):
        os.makedirs(tensorboard_dir)

except:
    sys.exit("Could not create model directory.")

# Save hyperparameter settings as json file
with open(os.path.join(model_dir, 'hyperparameter.json'), 'w') as fp:
    json.dump(settings, fp)

################### DATA FETCH ##########################

data = util.fetch.csv()
data['title'] = data['title'].str.lower()
data = util.fetch.csv().drop_duplicates('title').sort_values(['clicks'])

##################### STEMMING ############################

from nltk.stem.snowball import SnowballStemmer
import re
from keras.preprocessing import sequence

stemmer = SnowballStemmer('finnish')

stemmed_titles = []

stem_count = {}

for title in data.title:
    stemmed_title = []
    sanitized_title = re.sub("[^a-z0-9]", " ", title)
    for word in sanitized_title.split():
        stemmed_word = stemmer.stem(word)
        if stemmed_word in stem_count:
            stem_count[stemmed_word] += 1
        else:
            stem_count[stemmed_word] = 1
        stemmed_title.append(stemmer.stem(word))
    stemmed_titles.append(stemmed_title)

data['stemmed_title'] = stemmed_titles

#################### STEM INDICE MAPPING #######################

filtered_words = {k: v for k, v in stem_count.items() if v > settings['word_min_count']}
word_indices = dict((w, i + 1) for i, w in enumerate(filtered_words))
indices_word = dict((i + 1, w) for i, w in enumerate(filtered_words))

print("total words : " + str(len(filtered_words.keys())))

def titles_to_indices(titles, word_indices, max_len):
    X = []
    for i, title in enumerate(titles):
        X.append([])
        for j, word in enumerate(title[-max_len:]):
            if word in word_indices:
                X[i].append(word_indices[word])
    padded_X = sequence.pad_sequences(X, maxlen=max_len)
    return padded_X

#################### SET SPLITTING ##########################

rows = len(data)

tops = data[floor(rows - rows * settings['ratio']):]
bottoms = data[:floor(rows * settings['ratio'])]

tops['label'] = 1
bottoms['label'] = 0

shuffled_tops = tops.sample(frac=1).reset_index(drop=True)
shuffled_bottoms = bottoms.sample(frac=1).reset_index(drop=True)

test_index = len(tops) - len(tops) // 5
val_index = test_index - test_index // 5

train_top = shuffled_tops[:val_index]
val_top = shuffled_tops[val_index:test_index]
test_top = shuffled_tops[test_index:]

train_bottom = shuffled_bottoms[:val_index]
val_bottom = shuffled_bottoms[val_index:test_index]
test_bottom = shuffled_bottoms[test_index:]

test_data = pd.concat([test_top, test_bottom])
val_data = pd.concat([val_top, val_bottom])
train_data = pd.concat([train_top, train_bottom])

labeled_data = pd.concat([test_data, train_data, val_data])

titles = labeled_data['title']
labels = labeled_data['label']

####################### SET PROCESSING ###################

X_train = titles_to_indices(train_data['stemmed_title'], word_indices, settings['title_max_len'])
X_val = titles_to_indices(val_data['stemmed_title'], word_indices, settings['title_max_len'])
X_test = titles_to_indices(test_data['stemmed_title'], word_indices, settings['title_max_len'])
y_train = np.array(train_data['label'])
y_val = np.array(val_data['label'])
y_test = np.array(test_data['label'])

######################### MODEL ############################

# data set validation
print("Training data positive labels: %.2f%%" % (sum(y_train) / len(y_train) * 100))
print("Validation data positive labels: %.2f%%" % (sum(y_val) / len(y_val) * 100))
print("Testing data positive labels: %.2f%%" % (sum(y_test) / len(y_test) * 100))
print('Sample chars in X:{}'.format(X_train[12]))
print('y:{}'.format(y_train[12]))

# the model
model = Sequential()
model.add(Embedding(len(filtered_words) + 2, 1, input_length=settings['title_max_len'], mask_zero=True))
# model.add(LSTM(16, return_sequences=True))
# model.add(Dropout(0.3))
# model.add(LSTM(16, return_sequences=True))
# model.add(Dropout(0.1))
# model.add(LSTM(1))
model.add((LSTM(settings['lstm'])))
model.add(Dropout(settings['dropout']))
# model.add(Bidirectional(LSTM(16, return_sequences=True)))
# model.add(Dropout(0.2))
# model.add(Bidirectional(LSTM(16, return_sequences=True)))
# model.add(Dropout(0.2))
# model.add(LSTM(16))
# model.add(Dropout(0.3))
# model.add(Dense(2, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# callbacks
# learning rate reduction
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=1,
    min_lr=0.001
)
# checkpoint saving
checkpointer = ModelCheckpoint(
    model_dir+'/'+file_name+'.{epoch:02d}-{val_loss:.2f}.hdf5',
    monitor='val_loss',
    verbose=0,
    save_best_only=True,
    mode='min'
)
# early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
# batch history
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accs = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accs.append(logs.get('acc'))
batch_history = LossHistory()

# fit model
epoch_history = model.fit(
    X_train,
    y_train,
    nb_epoch=settings['epochs'],
    batch_size=settings['batch_size'],
    shuffle=True,
    validation_data=(X_val, y_val),
    callbacks=[checkpointer, batch_history, TensorBoard(log_dir='/tmp/rnn/' + date_string)]
)
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# confusion matrix
y_predict = model.predict_classes(X_test)
conf_matrix = confusion_matrix(y_test, y_predict)
print("\nConfusion matrix: ")
print(conf_matrix)

# plot model
#layer = model.layers[-2]
#f = K.function([model.layers[0].input, K.learning_phase()],
#               [layer.output])
#dense_layer_data = pandas.DataFrame(f([X_val, 0])[0])
#dense_layer_data[3] = model.predict(X_val)

#util.plot.dense_layer(dense_layer_data, model_dir)
util.plot.epochs(epoch_history, model_dir)
util.plot.batches(batch_history, model_dir)
