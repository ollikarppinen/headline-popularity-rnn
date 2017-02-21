#encoding: utf-8

import pandas
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils.data_utils import get_file
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
import numpy
from sklearn.metrics import confusion_matrix
import sys
import os

def fetch_csv():
    csv = pandas.read_csv("http://207.154.192.240/ampparit/ampparit.csv")
    csv['title'] = csv['title'].str.lower()
    return csv

data = fetch_csv().drop_duplicates('title')
sorted_data = data.sort_values(['clicks'])

rows = len(sorted_data)

ratio = 5
tops = sorted_data[(rows - rows // ratio):]
bottoms = sorted_data[:(rows // ratio)]

tops['label'] = 1
bottoms['label'] = 0

shuffled_tops = tops.sample(frac=1)
shuffled_bottoms = bottoms.sample(frac=1)

test_index = len(tops) - len(tops) // 5
val_index = test_index - test_index // 5

train_top = shuffled_tops[:val_index]
val_top = shuffled_tops[val_index:test_index]
test_top = shuffled_tops[test_index:]

train_bottom = shuffled_bottoms[:val_index]
val_bottom = shuffled_bottoms[val_index:test_index]
test_bottom = shuffled_bottoms[test_index:]

test_data = pandas.concat([test_top, test_bottom])
val_data = pandas.concat([val_top, val_bottom])
train_data = pandas.concat([train_top, train_bottom])

labeled_data = pandas.concat([test_data, train_data, val_data])

titles = labeled_data['title']
labels = labeled_data['label']

chars = {}

for title in titles:
    for char in title:
        if char in chars:
            chars[char] += 1
        else:
            chars[char] = 1

char_count_threshold = 100
chars = {k: v for k, v in chars.items() if v > char_count_threshold }
print('total chars:', len(chars))

char_indices = dict((c, i + 1) for i, c in enumerate(chars))
indices_char = dict((i + 1, c) for i, c in enumerate(chars))

title_max_len = 128

def strings_to_indices(strings, char_indices, max_len):
    X = []
    for i, string in enumerate(strings):
        X.append([])
        for j, char in enumerate(string[-max_len:]):
            if char in char_indices:
                X[i].append(char_indices[char])
    padded_X = sequence.pad_sequences(X, maxlen=max_len)
    return padded_X

X_train = strings_to_indices(train_data['title'], char_indices, title_max_len)
X_val = strings_to_indices(val_data['title'], char_indices, title_max_len)
X_test = strings_to_indices(test_data['title'], char_indices, title_max_len)
y_train = numpy.array(train_data['label'])
y_val = numpy.array(val_data['label'])
y_test = numpy.array(test_data['label'])

print("Training data positive labels: %.2f%%" % (sum(y_train) / len(y_train) * 100))
print("Validation data positive labels: %.2f%%" % (sum(y_val) / len(y_val) * 100))
print("Testing data positive labels: %.2f%%" % (sum(y_test) / len(y_test) * 100))
print('Sample chars in X:{}'.format(X_train[12]))
print('y:{}'.format(y_train[12]))

model = Sequential()
model.add(Embedding(len(chars) + 1, 1, input_length=title_max_len))
model.add(LSTM(20))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

file_name = os.path.basename(sys.argv[0]).split('.')[0]
checkpointer = ModelCheckpoint(
    'checkpoints/'+file_name+'.{epoch:02d}-{val_loss:.2f}.hdf5',
    monitor='val_loss',
    verbose=0,
    save_best_only=True,
    mode='min'
)
early_stopping = EarlyStopping(monitor='val_loss', patience=1)
model.fit(
    X_train,
    y_train,
    nb_epoch=100,
    batch_size=32,
    shuffle=True,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, TensorBoard(log_dir='/tmp/rnn'), checkpointer]
)

scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

y_predict = model.predict_classes(X_test)
conf_matrix = confusion_matrix(y_test, y_predict)
print("\nConfusion matrix: ")
print(conf_matrix)
