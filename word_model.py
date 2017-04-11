# -*- coding: utf-8 -*-

import numpy as np
np.random.seed(1) # Set random seed for Keras
import pandas as pd
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
    'ratio': 0.01,
    'title_max_len': 16,
    'batch_size': 64,
    'epochs': 1,
    'word_min_count': 20,
    'lstm': 16,
    'dropout': 0.5,
    'description': 'lstm 64 dropout 0.5',
    'embedding': 16,
    'early_stopping_patience': 2
}
########################### DIRECTORIES ########################################

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
    return padded_X.tolist()

#################### SET SPLITTING ##########################

rows = len(data)

tops = data[floor(rows - rows * settings['ratio']):]
bottoms = data[:floor(rows * settings['ratio'])]

tops['label'] = 1
bottoms['label'] = 0

shuffled_tops = tops.sample(frac=1, random_state=1).reset_index(drop=True)
shuffled_bottoms = bottoms.sample(frac=1, random_state=1).reset_index(drop=True)

kvantile_size = len(tops)

test_index = kvantile_size - (kvantile_size // 5) * 2
val_index = kvantile_size - kvantile_size // 5

train_top = shuffled_tops[:test_index]
val_top = shuffled_tops[test_index:val_index]
test_top = shuffled_tops[val_index:]

train_bottom = shuffled_bottoms[:test_index]
val_bottom = shuffled_bottoms[test_index:val_index]
test_bottom = shuffled_bottoms[val_index:]

test_data = pd.concat([test_top, test_bottom])
val_data = pd.concat([val_top, val_bottom])
train_data = pd.concat([train_top, train_bottom])

labeled_data = pd.concat([test_data, train_data, val_data])

titles = labeled_data['title']
labels = labeled_data['label']

####################### SET PROCESSING ###################

train_data['x'] = titles_to_indices(train_data['stemmed_title'], word_indices, settings['title_max_len'])
val_data['x'] = titles_to_indices(val_data['stemmed_title'], word_indices, settings['title_max_len'])
test_data['x'] = titles_to_indices(test_data['stemmed_title'], word_indices, settings['title_max_len'])
train_data['y'] = np.array(train_data['label'])
val_data['y'] = np.array(val_data['label'])
test_data['y'] = np.array(test_data['label'])

######################### DATASET VALIDATION ###################################

print("Training data positive labels: %.2f%%" % (sum(train_data.y) / len(train_data.y) * 100))
print("Validation data positive labels: %.2f%%" % (sum(val_data.y) / len(val_data.y) * 100))
print("Testing data positive labels: %.2f%%" % (sum(test_data.y) / len(test_data.y) * 100))
print('Sample chars in X:{}'.format(train_data.x.tolist()[12]))
print('y:{}'.format(train_data.y.tolist()[12]))

######################### MODEL ############################

model = Sequential()
model.add(Embedding(len(filtered_words) + 1, settings['embedding'], input_length=settings['title_max_len'], mask_zero=True))
model.add((LSTM(settings['lstm'])))
model.add(Dropout(settings['dropout']))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

######################### Callbacks ############################################

# Checkpoint saving
checkpointer = ModelCheckpoint(
    model_dir+'/'+file_name+'.{epoch:02d}-{val_loss:.2f}.hdf5',
    monitor='val_loss',
    verbose=0,
    save_best_only=True,
    mode='min'
)

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=settings['early_stopping_patience'])

# Batch history
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accs = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accs.append(logs.get('acc'))
batch_history = LossHistory()

# Fitting
epoch_history = model.fit(
    train_data.x.tolist(),
    train_data.y.tolist(),
    nb_epoch=settings['epochs'],
    batch_size=settings['batch_size'],
    shuffle=True,
    validation_data=(val_data.x.tolist(), val_data.y.tolist()),
    callbacks=[checkpointer, batch_history, TensorBoard(log_dir=os.path.join(tensorboard_dir, date_string)), early_stopping]
)
scores = model.evaluate(test_data.x.tolist(), test_data.y.tolist(), verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# Confusion matrix
y_predict = model.predict_classes(test_data.x.tolist())
conf_matrix = confusion_matrix(test_data.y.tolist(), y_predict)
print("\nConfusion matrix: ")
print(conf_matrix)

# Model plots
util.plot.epochs(epoch_history, model_dir)
util.plot.batches(batch_history, model_dir)
