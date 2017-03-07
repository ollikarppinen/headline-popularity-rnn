#encoding: utf-8

import pandas
import numpy
import sys
import os
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils.data_utils import get_file
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import RMSprop
from sklearn.metrics import confusion_matrix
from datetime import datetime
from math import floor
import matplotlib.pyplot as plt

def fetch_csv():
    csv = pandas.read_csv("http://207.154.192.240/ampparit/ampparit.csv")
    csv['title'] = csv['title'].str.lower()
    return csv

data = fetch_csv().drop_duplicates('title')
sorted_data = data.sort_values(['clicks'])

rows = len(sorted_data)

ratio = 0.01
tops = sorted_data[floor(rows - rows * ratio):]
bottoms = sorted_data[:floor(rows * ratio)]

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

# char indice mapping
chars = {}
for title in titles:
    for char in title:
        if char in chars:
            chars[char] += 1
        elif char.isalpha() or char == ' ' or char.isdigit():
            chars[char] = 1
char_count_threshold = 1
chars = {k: v for k, v in chars.items() if v > char_count_threshold }
max_features = len(chars)
print('total chars:', max_features)
char_indices = dict((c, i + 1) for i, c in enumerate(chars))
indices_char = dict((i + 1, c) for i, c in enumerate(chars))

title_max_len = 128

# strings to indices
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

# data set validation
print("Training data positive labels: %.2f%%" % (sum(y_train) / len(y_train) * 100))
print("Validation data positive labels: %.2f%%" % (sum(y_val) / len(y_val) * 100))
print("Testing data positive labels: %.2f%%" % (sum(y_test) / len(y_test) * 100))
print('Sample chars in X:{}'.format(X_train[12]))
print('y:{}'.format(y_train[12]))

# the model
model = Sequential()
model.add(Embedding(len(chars) + 1, 1, input_length=title_max_len))
# model.add(Convolution1D(128, activation='relu', filter_length=128))
# model.add(Convolution1D(128, activation='relu', filter_length=128))
# model.add(Convolution1D(128, activation='relu', filter_length=128))
# model.add(MaxPooling1D(64))
model.add(LSTM(64, return_sequences=True))
# model.add(Dropout(0.5))
# model.add(LSTM(64, return_sequences=True))
# model.add(Dropout(0.5))
model.add(LSTM(64))
# model.add(Dropout(0.5))
model.add(Dense(2, activation='relu'))
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
ts = datetime.now()
date_string = '-'.join(list(map(str, [ts.year, ts.month, ts.day, ts.hour, ts.minute])))
file_name = os.path.basename(sys.argv[0]).split('.')[0]
cwd = os.getcwd()
model_dir = os.path.join(cwd, 'checkpoints', date_string)
try:
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
except:
    sys.exit("Could not create model directory.")
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
    nb_epoch=10,
    batch_size=32,
    shuffle=True,
    validation_data=(X_val, y_val),
    callbacks=[checkpointer, batch_history]#, early_stopping, TensorBoard(log_dir='/tmp/rnn'), , reduce_lr]
)
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# confusion matrix
y_predict = model.predict_classes(X_test)
conf_matrix = confusion_matrix(y_test, y_predict)
print("\nConfusion matrix: ")
print(conf_matrix)

# plot model
layer = model.layers[-2]
f = K.function([model.layers[0].input, K.learning_phase()],
               [layer.output])
plot_data = pandas.DataFrame(f([X_val, 0])[0])
x = plot_data[0]
y = plot_data[1]
c = model.predict(X_val)
# dense layer scatter plot
plt.scatter(x,y,c=c,s=1, cmap='cool')
plt.savefig(os.path.join(model_dir, 'scatter.png'))
plt.clf()
# dense layer histogram
plt.hist2d(x,y,bins=200)
plt.savefig(os.path.join(model_dir, 'hist.png'))
plt.clf()
# accuracy history
plt.plot(epoch_history.history['acc'])
plt.plot(epoch_history.history['val_acc'])
plt.title('model epoch accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig(os.path.join(model_dir, 'epoch_accuracy.png'))
plt.clf()
# loss history
plt.plot(epoch_history.history['loss'])
plt.plot(epoch_history.history['val_loss'])
plt.title('model epoch loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig(os.path.join(model_dir, 'epoch_loss.png'))
# batch testing accuracy history
plt.plot(batch_history.accs)
plt.title('model batch accuracy')
plt.ylabel('accuracy')
plt.xlabel('batch')
plt.legend(['train'], loc='upper left')
plt.savefig(os.path.join(model_dir, 'batch_accuracy.png'))
plt.clf()
# batch testing loss history
plt.plot(batch_history.losses)
plt.title('model batch loss')
plt.ylabel('loss')
plt.xlabel('batch')
plt.legend(['train'], loc='upper left')
plt.savefig(os.path.join(model_dir, 'batch_loss.png'))
